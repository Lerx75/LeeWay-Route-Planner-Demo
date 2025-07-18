import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import requests

st.title("LeeWay v3.1 — Road Distance Territory call Clustering")

st.markdown("""
Upload your calls (with postcode), pick min/max calls per day.
Clusters will be assigned by real-world **road distance**.
""")

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
dfm = pd.merge(df, lookup[["Postcode", "Latitude", "Longitude"]], left_on="postcode_clean", right_on="Postcode", how="left")
df_geocoded = dfm.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True).copy()
df_failed = dfm[dfm["Latitude"].isna()].reset_index(drop=True).copy()
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

with st.form("params"):
    min_calls = st.number_input("Min calls per territory", 1, 100, 5)
    max_calls = st.number_input("Max calls per territory", int(min_calls), 100, int(min_calls)+1)
    run = st.form_submit_button("Run")

if not run:
    st.stop()

coords = df_geocoded[["Latitude", "Longitude"]].to_numpy()
n_calls = len(coords)

# --- OSRM batching for true road distances ---
import os

# Use an environment variable if set, otherwise fall back to public OSRM
osrm_url = os.getenv("OSRM_URL", "https://router.project-osrm.org")

def osrm_table_batch(coords, batch_size=100):
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(0, n, batch_size):
        from_idx = list(range(i, min(i+batch_size, n)))
        loc_from = coords[from_idx]
        for j in range(0, n, batch_size):
            to_idx = list(range(j, min(j+batch_size, n)))
            loc_to = coords[to_idx]
            locs = [f"{lon},{lat}" for lat, lon in np.vstack([loc_from, loc_to])]
            srcs = list(range(len(loc_from)))
            dsts = list(range(len(loc_from), len(loc_from) + len(loc_to)))
            req = f"{osrm_url}/table/v1/driving/" + ";".join(locs)
            params = {
                "sources": ";".join(map(str, srcs)),
                "destinations": ";".join(map(str, dsts)),
                "annotations": "distance"
            }
            r = requests.get(req, params=params)
            data = r.json()
            if r.status_code == 200 and "distances" in data:
                chunk = np.array(data["distances"])
                matrix[np.ix_(from_idx, to_idx)] = chunk
            else:
                st.warning(f"OSRM cross-batch error: {r.text}")
                # fallback to Haversine
                from_pts = coords[from_idx]
                to_pts = coords[to_idx]
                for a, ai in enumerate(from_idx):
                    for b, bi in enumerate(to_idx):
                        lat1, lon1 = from_pts[a]
                        lat2, lon2 = to_pts[b]
                        R = 6371000
                        phi1, phi2 = np.radians(lat1), np.radians(lat2)
                        dphi = phi2 - phi1
                        dlambda = np.radians(lon2 - lon1)
                        aa = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
                        dist = int(R * 2 * np.arctan2(np.sqrt(aa), np.sqrt(1-aa)))
                        matrix[ai, bi] = dist
    return matrix

with st.spinner("Calculating road distances..."):
    dist_matrix = osrm_table_batch(coords)

# --- Greedy road distance clustering ---
def greedy_cluster(dist_matrix, min_calls, max_calls):
    n = dist_matrix.shape[0]
    unassigned = set(range(n))
    groups = []
    while len(unassigned) >= min_calls:
        # Pick the most isolated (max min distance to others)
        dists = dist_matrix[list(unassigned)][:, list(unassigned)]
        if len(unassigned) == n:
            # First group: pick any
            seed = list(unassigned)[0]
        else:
            row_min = dists + np.eye(len(dists)) * 1e12
            farthest = np.argmax(np.min(row_min, axis=1))
            seed = list(unassigned)[farthest]
        # Find nearest calls to seed
        dists_to_seed = dist_matrix[seed, list(unassigned)]
        nearest_idx = np.argsort(dists_to_seed)[:max_calls]
        group = [list(unassigned)[i] for i in nearest_idx[:min_calls]]
        # Try to expand group up to max_calls if all close
        for i in nearest_idx[min_calls:]:
            if len(group) >= max_calls:
                break
            # Optional: only add if not stretching too far
            if dists_to_seed[i] < 2 * np.mean(dists_to_seed[nearest_idx[:min_calls]]):
                group.append(list(unassigned)[i])
        groups.append(group)
        unassigned -= set(group)
    # Leftovers: assign to nearest existing group (if >0 left)
    if unassigned:
        for i in list(unassigned):
            # Assign to group with lowest avg road distance
            best_group = min(range(len(groups)), key=lambda g: np.mean(dist_matrix[i, groups[g]]))
            groups[best_group].append(i)
        unassigned = set()
    return groups

with st.spinner("Optimizing cluster assignment by road distance (this may take several minutes for 500+ calls)..."):
    groups = greedy_cluster(dist_matrix, int(min_calls), int(max_calls))

# --- Flag any group that's too small ---
group_sizes = [len(g) for g in groups]
sparse_group_index = None
for idx, sz in enumerate(group_sizes):
    if sz < int(min_calls):
        sparse_group_index = idx
        break

sparse_flags = []
for idx, group in enumerate(groups):
    flag = "YES" if idx == sparse_group_index else ""
    for call in group:
        sparse_flags.append(flag)

# --- Output: Add group assignment and flags ---
group_col = []
for gnum, group in enumerate(groups, 1):
    for idx in group:
        group_col.append(f"Day {gnum}")

df_geocoded = df_geocoded.iloc[[i for group in groups for i in group]].copy()
df_geocoded["Group"] = group_col
df_geocoded["DayNum"] = df_geocoded["Group"].str.extract(r'(\d+)').astype(int)
df_geocoded["CallsInDay"] = df_geocoded.groupby("Group")["Group"].transform("count")
df_geocoded["SparseWarning"] = sparse_flags

cols_out = [postcode_col, "Latitude", "Longitude", "Group", "DayNum", "CallsInDay", "SparseWarning"] + \
           [c for c in df_geocoded.columns if c not in [postcode_col, "Latitude", "Longitude", "Group", "DayNum", "CallsInDay", "postcode_clean", "Postcode", "SparseWarning"]]

st.subheader("Resulting Clusters (table)")
st.dataframe(df_geocoded[cols_out], height=400)

buf = io.BytesIO()
df_geocoded[cols_out].to_excel(buf, index=False, engine="openpyxl")
buf.seek(0)
st.download_button(
    "Download grouped territories as Excel",
    data=buf,
    file_name="grouped_customer_routes.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


