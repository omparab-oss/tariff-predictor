import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import re
import requests
try:
    from streamlit_searchbox import st_searchbox
    HAS_SEARCHBOX = True
except Exception:
    HAS_SEARCHBOX = False

# Ensure objects saved from notebooks under module name 'main' can be resolved here
sys.modules['main'] = sys.modules.get('main', sys.modules[__name__])

# ---------- Compatibility: functions/classes referenced by the saved pipeline ----------
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if 'miles' in X.columns and 'tariff' in X.columns:
        denom = X['miles'].replace(0, np.nan)
        X['tariff_per_mile'] = X['tariff'] / denom
    if 'miles' in X.columns and 'gp' in X.columns:
        denom = X['miles'].replace(0, np.nan)
        X['gp_per_mile'] = X['gp'] / denom
    if 'origin_state' in X.columns and 'destination_state' in X.columns:
        X['is_same_state'] = (
            X['origin_state'].astype(str).str.strip().str.lower()
            == X['destination_state'].astype(str).str.strip().str.lower()
        ).astype(int)
    else:
        X['is_same_state'] = 0
    return X


class VehicleTypesBinarizer(BaseEstimator, TransformerMixin):
    """Multi-label binarizer for comma-separated vehicle_types strings."""
    def __init__(self, sep=','):
        self.sep = sep
        self.mlb = MultiLabelBinarizer()
        self.feature_names_ = None

    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                return X.iloc[:, 0]
            return X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            return X
        else:
            X = np.asarray(X)
            if X.ndim == 2 and X.shape[1] == 1:
                return pd.Series(X[:, 0])
            return pd.Series(X.ravel())

    def _tokenize(self, text):
        if not isinstance(text, str):
            return []
        return [t.strip().lower() for t in text.split(self.sep) if t.strip()]

    def fit(self, X, y=None):
        s = self._to_series(X).fillna('')
        labels = s.map(self._tokenize)
        self.mlb.fit(labels)
        self.feature_names_ = [f"vt__{t}" for t in self.mlb.classes_]
        return self

    def transform(self, X):
        s = self._to_series(X).fillna('')
        labels = s.map(self._tokenize)
        arr = self.mlb.transform(labels)
        return arr

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_) if self.feature_names_ is not None else np.array([])

APP_TITLE = "Carrier Pay Predictor"
MODEL_PATH = Path("models/carrier_pay_model_2.joblib")
METRICS_PATH = Path("models/carrier_pay_model_metrics_2.json")
HEADER_IMAGE = Path("data/image.png")

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------- Utilities ----------

def load_image(path: Path):
    try:
        if path.exists():
            return Image.open(path)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def geocode_search(query: str, limit: int = 10):
    """Query Nominatim for city/state suggestions."""
    q = (query or "").strip()
    if len(q) < 3:
        return []
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"format": "json", "addressdetails": 1, "limit": limit, "q": q}
        headers = {"User-Agent": "carrier-pay-app/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        for item in data:
            addr = item.get("address", {})
            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet") or addr.get("county")
            state = addr.get("state") or addr.get("region")
            country = addr.get("country")
            try:
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
            except Exception:
                continue
            if city and state:
                label = f"{city}, {state}{', ' + country if country else ''}"
                results.append({
                    "label": label,
                    "city": str(city),
                    "state": str(state),
                    "country": str(country) if country else "",
                    "lat": lat,
                    "lon": lon,
                })
        return results
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def osrm_distance_miles(origin: dict, destination: dict):
    """Use OSRM to compute driving distance in miles between two lat/lon points."""
    try:
        if not origin or not destination:
            return None
        lon1, lat1 = origin["lon"], origin["lat"]
        lon2, lat2 = destination["lon"], destination["lat"]
        url = f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
        params = {"overview": "false", "alternatives": "false", "annotations": "false"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        routes = data.get("routes")
        if not routes:
            return None
        meters = routes[0].get("distance")
        if meters is None:
            return None
        return meters / 1609.344
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def extract_options_from_model(_pipeline):
    """Extract categorical options learned by the model (for autocomplete dropdowns)."""
    options = {}
    try:
        preprocess = _pipeline.named_steps.get("preprocess")
        if preprocess is None:
            return options
        # Fitted transformers
        for name, transformer, cols in getattr(preprocess, "transformers_", []):
            if name == "cat":
                # Pipeline(imputer, onehot)
                try:
                    ohe = transformer.named_steps.get("onehot")
                    if ohe is not None and hasattr(ohe, "categories_"):
                        for col, cats in zip(cols, ohe.categories_):
                            options[col] = [str(c) for c in cats]
                except Exception:
                    pass
            elif name == "vtypes":
                # Custom VehicleTypesBinarizer pipeline
                try:
                    vt_bin = transformer.named_steps.get("vt_bin")
                    if vt_bin is not None and hasattr(vt_bin, "mlb") and hasattr(vt_bin.mlb, "classes_"):
                        options["vehicle_types"] = [str(c) for c in vt_bin.mlb.classes_]
                except Exception:
                    pass
    except Exception:
        pass
    return options


def comma_join(values):
    if not values:
        return ""
    return ", ".join([str(v).strip() for v in values if str(v).strip()])


def predict_carrier_pay(pipeline, payload: dict):
    df = pd.DataFrame([payload])
    preds = pipeline.predict(df)
    return float(preds[0])


# ---------- UI ----------

# (Header image removed per request)

st.markdown(f"### {APP_TITLE}")

# Load model & metadata
try:
    pipe = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

metrics = load_metrics() or {}
options = extract_options_from_model(pipe)

# Column layout: left = inputs, right = results
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Inputs")
    # 1) Locations (outside the form) — realtime autocomplete with loading indicators
    st.markdown("#### Locations")
    from_place = st.session_state.get('from_place')
    to_place = st.session_state.get('to_place')

    col_from, col_to = st.columns(2)
    with col_from:
        from_query = st.text_input("From: city, state", key="from_query")
        if st.button("Search From", key="search_from_btn"):
            with st.spinner("Searching From…"):
                st.session_state['from_opts'] = geocode_search(st.session_state.get('from_query', ''))
        from_opts = st.session_state.get('from_opts', [])
        if from_opts:
            from_label = st.selectbox("Select From", options=[o['label'] for o in from_opts], key='from_sel')
            from_place = next((o for o in from_opts if o['label'] == from_label), None)
            st.session_state['from_place'] = from_place
            if from_place:
                st.session_state['origin_city'] = from_place['city']
                st.session_state['origin_state'] = from_place['state']
    with col_to:
        to_query = st.text_input("To: city, state", key="to_query")
        if st.button("Search To", key="search_to_btn"):
            with st.spinner("Searching To…"):
                st.session_state['to_opts'] = geocode_search(st.session_state.get('to_query', ''))
        to_opts = st.session_state.get('to_opts', [])
        if to_opts:
            to_label = st.selectbox("Select To", options=[o['label'] for o in to_opts], key='to_sel')
            to_place = next((o for o in to_opts if o['label'] == to_label), None)
            st.session_state['to_place'] = to_place
            if to_place:
                st.session_state['destination_city'] = to_place['city']
                st.session_state['destination_state'] = to_place['state']

    miles = None
    if from_place and to_place:
        with st.spinner("Computing distance…"):
            miles = osrm_distance_miles(from_place, to_place)
        st.session_state['miles'] = miles
        if miles:
            st.success(f"Distance: {miles:,.1f} miles")
        else:
            st.warning("Could not compute distance between selected locations.")

    # Quick summary row for selections
    sum_c1, sum_c2, sum_c3 = st.columns(3)
    with sum_c1:
        st.caption("From")
        st.write(f"{st.session_state.get('origin_city','')}, {st.session_state.get('origin_state','')}" if st.session_state.get('origin_city') else "-")
    with sum_c2:
        st.caption("To")
        st.write(f"{st.session_state.get('destination_city','')}, {st.session_state.get('destination_state','')}" if st.session_state.get('destination_city') else "-")
    with sum_c3:
        st.caption("Distance (mi)")
        st.write(f"{st.session_state.get('miles'):.1f}" if isinstance(st.session_state.get('miles'), (int, float)) else "-")

    with st.form("input_form"):

        # 2) Mode (no customer field)
        mode_opts = options.get("mode", ["Open", "Enclosed"])
        mode = st.selectbox("Mode", options=mode_opts, key="mode")

        # Default customer from model categories (no user input)
        cust_opts = options.get("customer") or options.get("Customer") or []
        customer = cust_opts[0] if cust_opts else ""

        # 3) Vehicle management: add one-by-one with Inop flag, build model input tokens
        vt_vocab = options.get("vehicle_types", options.get("Vehicle_Types", []))
        vt_vocab = [str(v).strip() for v in (vt_vocab or []) if str(v).strip()]
        def _normalize_base(tok: str) -> str:
            t = str(tok).lower().strip()
            t = re.sub(r"\s*\(.*?\)\s*", " ", t)  # remove parenthetical flags
            t = re.sub(r"\b(inop|op)\b", "", t)     # remove op/inop words
            t = t.replace("/", " ")                    # normalize separators
            t = t.replace("-", " ")
            return " ".join(t.split())
        base_set = set()
        for raw in vt_vocab:
            parts = [p.strip() for p in str(raw).split(",") if p.strip()]
            for p in parts:
                n = _normalize_base(p)
                if n:
                    base_set.add(n)
        if not base_set:
            base_set = {"sedan", "suv", "pickup", "van"}
        # Title case for display; mapping to model remains case-insensitive
        base_types = sorted({b.title() for b in base_set})
        if 'vehicles' not in st.session_state:
            st.session_state['vehicles'] = []

        col_v1, col_v2, col_v3, col_v4 = st.columns([2, 1, 1, 1])
        with col_v1:
            vtype_sel = st.selectbox("Vehicle Type", options=base_types, key="vtype_sel")
        with col_v2:
            v_inop = st.checkbox("Inop", key="v_inop", value=False)
        with col_v3:
            add_vehicle = st.form_submit_button("Add")
        with col_v4:
            remove_last = st.form_submit_button("Remove")

        if add_vehicle and vtype_sel:
            st.session_state['vehicles'].append({"type": vtype_sel, "inop": bool(v_inop)})
        if remove_last and st.session_state['vehicles']:
            st.session_state['vehicles'].pop()

        # Show bracketed tokens for current vehicles
        display_tokens = [f"{v['type']}({'inop' if v['inop'] else 'op'})" for v in st.session_state['vehicles']]
        if st.session_state['vehicles']:
            st.markdown("Current vehicles:")
            st.write(", ".join(display_tokens))

        vocab_set = set([t.lower().strip() for t in vt_vocab])
        def _to_model_token(base: str, inop: bool) -> str:
            """Map chosen base+op/inop to a known vocab token if available; else bracket style without space."""
            base = base.lower().strip()
            flag = "inop" if inop else "op"
            candidates = [f"{base} ({flag})", f"{base}({flag})", base]
            for c in candidates:
                if c in vocab_set:
                    return c
            return f"{base}({flag})"
        model_tokens = [_to_model_token(v['type'], v['inop']) for v in st.session_state['vehicles']]
        vehicle_types = ", ".join(model_tokens)

        # Build location fields from session (selected above)
        origin_city = st.session_state.get("origin_city", "")
        origin_state = st.session_state.get("origin_state", "")
        destination_city = st.session_state.get("destination_city", "")
        destination_state = st.session_state.get("destination_state", "")

        total_vehicles = len(st.session_state['vehicles'])
        st.info(f"Total Vehicles: {total_vehicles}")
        # No tariff/GP inputs; set as NaN for pipeline imputers. We'll compute for display after prediction.
        tariff = float('nan')
        gp = float('nan')
        # Read computed miles from session
        miles = st.session_state.get('miles')

        submitted = st.form_submit_button("Predict Carrier Pay", use_container_width=True)

    # Build payload now so right column can show a preview
    payload = {
        "customer": customer,
        "vehicle_types": vehicle_types,
        "mode": mode,
        "origin_city": origin_city,
        "destination_city": destination_city,
        "origin_state": origin_state,
        "destination_state": destination_state,
        "miles": miles,
        "tariff": tariff,
        "gp": gp,
        "total_vehicles": total_vehicles,
    }

with right:
    st.subheader("Prediction")
    if metrics:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Model", str(metrics.get('best_base_model','?')).upper())
        with m2:
            _rmse = metrics.get('best_cv_rmse', None)
            st.metric("CV RMSE", f"{_rmse:.2f}" if isinstance(_rmse, (int, float)) else "?")
        with m3:
            _r2 = metrics.get('final_r2', None)
            st.metric("R²", f"{_r2:.3f}" if isinstance(_r2, (int, float)) else "?")

    with st.expander("Input preview (without Tariff/GP)", expanded=False):
        _preview = {k: v for k, v in payload.items() if k not in ("tariff", "gp")}
        st.dataframe(pd.DataFrame([_preview]), use_container_width=True)

    if 'submitted' in locals() and submitted:
        if total_vehicles < 1:
            st.error("Please add at least one vehicle before predicting.")
        elif not (origin_city and origin_state and destination_city and destination_state and miles and miles > 0):
            st.error("Please select valid From and To locations (distance must be computed).")
        else:
            try:
                pred = predict_carrier_pay(pipe, payload)
                st.success(f"Estimated Carrier Pay: ${pred:,.2f}")
                # Compute tariff and gp based on rule: GP = 30% of Tariff, Tariff = Carrier Pay + GP => Tariff = CP / 0.7
                est_tariff = pred / 0.7
                est_gp = 0.3 * est_tariff
                tcol, gcol = st.columns(2)
                with tcol:
                    st.info(f"Tariff: ${est_tariff:,.2f}")
                with gcol:
                    st.info(f"GP (30%): ${est_gp:,.2f}")
                if miles and miles > 0:
                    st.caption(f"Tariff/mile: ${est_tariff/miles:,.2f} | GP/mile: ${est_gp/miles:,.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Tip: Dropdowns are searchable. Start typing to autocomplete.")
