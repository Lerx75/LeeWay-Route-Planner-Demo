import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

st.title("LeeWay v3.1 — Journey planner")

st.markdown("""
Upload your calls (with postcode), pick min/max calls per day.
Clusters will be assigned by real-world **road distance**.
""")

# --- File upload & geocoding setup ---
uploaded = st.file_uploader("Upload Excel (.xlsx) with a postcode column", type="xlsx")
if not uploaded:
    st.stop()

df = pd.read_excel(uploaded).convert_dtypes()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")
postcode_col = next((c for c in df.columns if c in ["postcode", "post_code"]), None)
if not postcode_col:
    st.error("Your file needs a column named `postcode` or `post_code`.")
    st.stop()

lookup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postcode_lookup.csv")
try:
    lookup = pd.read_csv(lookup_path)
except FileNotFoundError:
    st.error("postcode_lookup.csv not found.")
    st.stop()

lookup["Postcode"] = lookup["Postcode"].astype(str).str.replace(" ", "").str.upper()
df["postcode_clean"] = df[postcode_col].astype(str).str.replace(" ", "").str.upper()
dfm = pd.merge(
    df,
    lookup[["Postcode", "Latitude", "Longitude"]],
    left_on="postcode_clean",
    right_on="Postcode",
    how="left"
)
df_geocoded = dfm.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
df_failed = dfm[dfm["Latitude"].isna()].reset_index(drop=True)

if df_geocoded.empty:
    st.error("No valid coordinates—check your postcode_lookup.csv.")
    st.stop()

if not df_failed.empty:
    failbuf = io.BytesIO()
    df_failed[[postcode_col]].to_excel(failbuf, index=False, engine="openpyxl")
    failbuf.seek(0)
    st.warning(f"{len(df_failed)} postcodes could not be located and will not be included.")
    st.download_button("Download failed postcodes", failbuf, "failed_to_locate.xlsx")

st.success(f"{len(df_geocoded)} calls geocoded and ready.")

# --- Parameter form ---
with st.form("params"):
    min_calls = st.number_input("Min calls per territory", 1, 100, 5)
    max_calls = st.number_input("Max calls per territory", int(min_calls), 100, int(min_calls) + 1)
    run = st.form_submit_button("Run")

if not run:
    st.stop()

coords = df_geocoded[["Latitude", "Longitude"]].to_numpy()
n_calls = len(coords)

# --- Chunked + parallel OSRM distance matrix ---
osrm_url = os.getenv("OSRM_URL", "https://router.project-osrm.org")

def osrm_table_chunked(coords, osrm_url, chunk_size=100, workers=8):
    n = len(coords)
    dist = np.zeros((n, n), dtype=int)
    # split into blocks of up to `chunk_size`
    chunks = [coords[i : i + chunk_size] for i in range(0, n, chunk_size)]

    def fetch_block(i, j, from_chunk, to_chunk):
        # build the list of "lon,lat" strings
        locs = [f"{lon},{lat}" for lat, lon in np.vstack([from_chunk, to_chunk])]
        params = {
            "sources":      ";".join(map(str, range(len(from_chunk)))),
            "destinations": ";".join(map(str, range(len(from_chunk),
                                                    len(from_chunk) + len(to_chunk)))),
            "annotations":  "distance"
        }
        url = f"{osrm_url}/table/v1/driving/" + ";".join(locs)
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return i, j, np.array(r.json()["distances"], dtype=int)

    # dispatch all block-pairs in parallel
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        for i, fc in enumerate(chunks):
            for j, tc in enumerate(chunks):
                tasks.append(exe.submit(fetch_block, i, j, fc, tc))
        for fut in as_completed(tasks):
            i, j, block = fut.result()
            ro, co = i * chunk_size, j * chunk_size
            dist[ro : ro + block.shape[0], co : co + block.shape[1]] = block

    return dist

with st.spinner("Calculating road distances (chunked OSRM)…"):
    dist_matrix = osrm_table_chunked(coords, osrm_url, chunk_size=100, workers=8)

# --- Greedy road-distance clustering ---
def greedy_cluster(dist_matrix, min_calls, max_calls):
    n = dist_matrix.shape[0]
    unassigned = set(range(n))
    groups = []
    while len(unassigned) >= min_calls:
        dists = dist_matrix[list(unassigned)][:, list(unassigned)]
        if len(unassigned) == n:
            seed = list(unassigned)[0]
        else:
            row_min = dists + np.eye(len(dists)) * 1e12
            seed = list(unassigned)[np.argmax(np.min(row_min, axis=1))]

        d2s = dist_matrix[seed, list(unassigned)]
        nearest = np.argsort(d2s)
        group = [list(unassigned)[i] for i in nearest[:min_calls]]

        for i in nearest[min_calls:]:
            if len(group) >= max_calls:
                break
            if d2s[i] < 2 * np.mean(d2s[nearest[:min_calls]]):
                group.append(list(unassigned)[i])

        groups.append(group)
        unassigned -= set(group)

    if unassigned:
        for i in list(unassigned):
            best = min(range(len(groups)),
                       key=lambda g: np.mean(dist_matrix[i, groups[g]]))
            groups[best].append(i)
    return groups

with st.spinner("Optimizing cluster assignment…"):
    groups = greedy_cluster(dist_matrix, int(min_calls), int(max_calls))

# --- Flag too-small groups ---
group_sizes = [len(g) for g in groups]
sparse_idx = next((i for i, sz in enumerate(group_sizes) if sz < min_calls), None)
sparse_flags = [("YES" if idx == sparse_idx else "") for idx, grp in enumerate(groups) for _ in grp]

# --- Build output DataFrame ---
group_col = [f"Day {i+1}" for i, grp in enumerate(groups) for _ in grp]

df_out = df_geocoded.iloc[[i for grp in groups for i in grp]].copy()
df_out["Group"] = group_col
df_out["DayNum"] = df_out["Group"].str.extract(r'(\d+)').astype(int)
df_out["CallsInDay"] = df_out.groupby("Group")["Group"].transform("count")
df_out["SparseWarning"] = sparse_flags

cols_fixed = [postcode_col, "Latitude", "Longitude", "Group", "DayNum", "CallsInDay", "SparseWarning"]
other_cols = [c for c in df_out.columns if c not in cols_fixed]
cols_out = cols_fixed + other_cols

st.subheader("Resulting Clusters (table)")
st.dataframe(df_out[cols_out], height=400)

# --- Download button ---
buf = io.BytesIO()
df_out[cols_out].to_excel(buf, index=False, engine="openpyxl")
buf.seek(0)
st.download_button(
    "Download grouped territories as Excel",
    data=buf,
    file_name="grouped_customer_routes.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


