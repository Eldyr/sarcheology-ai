"""Offline expert review of a restricted, locally generated catalogue."""
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sarchaeology.candidates.review import DECISIONS,feedback,save_feedback
from sarchaeology.fusion.io import read_json,digest


@st.cache_data
def footprints(path,checksum,crs):
    if digest(path)!=checksum:
        raise ValueError("Catalogue checksum differs from its provenance")
    return gpd.read_file(path).to_crs(crs)


def local_map(frame,cid):
    fig = go.Figure()
    for selected,color,width in [(False,"#618790",1),(True,"#d35825",3)]:
        xs,ys = [],[]
        for geometry in frame.loc[(frame.candidate_id==cid)==selected,"geometry"]:
            for g in getattr(geometry,"geoms",[geometry]):
                x,y = g.exterior.xy
                xs.extend([*x,None])
                ys.extend([*y,None])
        fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line=dict(color=color,width=width),hoverinfo="skip"))
    fig.update_layout(showlegend=False,height=415,margin=dict(l=0,r=0,t=0,b=0),dragmode="pan",
                      xaxis_title="Easting (m)",yaxis_title="Northing (m)",yaxis_scaleanchor="x")
    return fig


def main():
    st.set_page_config(page_title="Candidate review",layout="wide")
    st.title("Candidate review")
    run = Path(os.environ.get("SARCHAEOLOGY_REVIEW_RUN","outputs/phase5/buto_2024"))
    if not (run/"provenance.json").is_file():
        st.info("Build the Phase 5 catalogue to open the review workspace.")
        return
    p = read_json(run/"provenance.json")
    h = p["catalogue_sha256"]
    cat = pd.read_csv(run/"ranked.csv")
    if cat.empty:
        st.info("No candidates meet the configured extraction rules.")
        return
    options = dict(zip(cat.candidate_id,[f"#{r['rank']}  ·  {r['candidate_id']}  ·  {r['priority']:.1f}" for r in cat.to_dict("records")]))
    cid = st.selectbox("Ranked candidate",list(options),format_func=options.get)
    record = read_json(run/"reports"/cid/"evidence.json")
    r = record["ranking"]
    cols = st.columns(4)
    for col,label,value in zip(cols,["Priority / 100","Area","Quarters observed","SAR dates"],
                               [f"{r['priority']:.1f}",f"{record['morphology']['area_m2']:,.0f} m²",r["available_quarters"],r["sar_dates"]]):
        col.metric(label,value)
    periods = record["profile"]["quarterly"]
    period = st.selectbox("Observation quarter",list(periods))
    st.image(str(run/"reports"/cid/f"{period}.png"),width="stretch")
    with st.expander("Footprints and projected coordinates"):
        from rasterio import open as open_raster
        with open_raster(run/"gis"/f"{period}_anomalies.tif") as src:
            crs = str(src.crs)
        st.plotly_chart(local_map(footprints(run/"gis/catalogue.geojson",h,crs),cid),width="stretch")
    with st.expander("Evidence and ranking",expanded=True):
        st.dataframe(pd.DataFrame([{"Term":k,"Priority points":round(v,2)} for k,v in r["contributions"].items()]))
        st.write(f"Screening penalty: {r['penalty_points']:.1f} points")
        st.write(", ".join(r["flags"]) or "No screening flags triggered.")
        st.json({"quarter":periods[period],"sources":record["context"],"terrain":record["profile"]["terrain"],
                 "acquisitions":record["profile"]["acquisitions"]},expanded=False)
    db = run/"feedback/expert.sqlite"
    reviewer = st.text_input("Reviewer identifier").strip()
    history,latest,agreement = feedback(db,h)
    previous = latest[(latest.candidate_id==cid)&(latest.reviewer==reviewer)]
    current = int(previous.revision.iloc[0]) if len(previous) else 0
    key = (h,cid,reviewer)
    if st.session_state.get("review_key")!=key:
        st.session_state.review_key = key
        st.session_state.review_revision = current
    with st.form(f"review-{cid}-{reviewer}"):
        decision = st.radio("Assessment",DECISIONS,format_func=lambda s:s.replace("_"," ").title(),
                            index=DECISIONS.index(previous.decision.iloc[0]) if len(previous) else 2)
        note = st.text_area("Reason and suggested follow-up",value=previous.note.iloc[0] if len(previous) else "")
        if st.form_submit_button("Save review",disabled=not reviewer):
            try:
                save_feedback(db,h,cid,reviewer,decision,note,st.session_state.review_revision,options)
                st.session_state.review_revision += 1
                st.success("Review saved.")
            except ValueError as exc:
                st.session_state.review_key = None
                st.error(str(exc))
    if st.checkbox("Show reviewer agreement"):
        _,_,agreement = feedback(db,h)
        if agreement:
            st.dataframe(pd.DataFrame(agreement))
        else:
            st.info("Agreement requires two reviewers to assess the same candidate.")


if __name__=="__main__":
    main()
