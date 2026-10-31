
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import io, math

st.set_page_config(page_title="JAL-DRISHTI", page_icon="💧", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.main-title{font-size:42px;font-weight:800;margin-bottom:0}
.subtitle{font-size:18px;color:#4b5563;margin-top:0}
.card{padding:18px;border-radius:14px;border:1px solid #dbe3ea;background:#f8fbfd}
.metric{font-size:28px;font-weight:750}
.small{font-size:13px;color:#667085}
</style>
""", unsafe_allow_html=True)

# ---------- Core prototype analysis ----------
def resize_image(img, max_side=420):
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1:
        img = img.resize((int(img.width*scale), int(img.height*scale)))
    return img

def analyse(img):
    a = np.asarray(img).astype(np.float32) / 255.0
    r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]

    # Prototype land-cover heuristics (replace with trained segmentation model later)
    green = (g > r*1.08) & (g > b*1.10) & (g > 0.22)
    water = (b > r*1.10) & (b > g*1.02) & (b > 0.28)
    bare = (~green) & (~water) & (((r + g + b) / 3) > 0.32) & (r > b * 1.05)
    built = (~green & ~water & ~bare)

    cls = np.zeros(r.shape, dtype=np.uint8)
    cls[green] = 1
    cls[water] = 2
    cls[bare] = 3
    cls[built] = 4

    # Synthetic terrain proxy from luminance gradients for demonstration
    lum = 0.299*r + 0.587*g + 0.114*b
    gx = np.gradient(lum, axis=1)
    gy = np.gradient(lum, axis=0)
    slope = np.sqrt(gx**2 + gy**2)
    slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)

    # Drainage proxy: darker/blue pixels become more likely drainage
    drainage = (water.astype(float) * 0.75 + (1-lum)*0.15 + slope*0.10)
    drainage = (drainage - drainage.min())/(drainage.max()-drainage.min()+1e-8)

    # Priority: bare soil + slope + low vegetation + proximity to drainage proxy
    priority = (0.40*bare.astype(float) + 0.28*slope + 0.18*(1-green.astype(float)) + 0.14*drainage)
    priority = (priority-priority.min())/(priority.max()-priority.min()+1e-8)

    # Exclude obvious water from intervention priority
    priority[water] *= 0.15

    # Top priority zones by quantile
    threshold = np.quantile(priority, 0.88)
    zones = priority >= threshold

    return cls, slope, drainage, priority, zones

def class_image(cls):
    # 0 built/other, 1 vegetation, 2 water, 3 bare soil, 4 built
    out = np.zeros((*cls.shape,3), dtype=np.uint8)
    out[cls==0] = [180,180,180]
    out[cls==1] = [50,170,80]
    out[cls==2] = [50,130,220]
    out[cls==3] = [205,155,75]
    out[cls==4] = [150,100,150]
    return Image.fromarray(out)

def heatmap(priority):
    p = (priority*255).clip(0,255).astype(np.uint8)
    # simple blue->yellow->red RGB ramp
    out=np.zeros((*p.shape,3),dtype=np.uint8)
    t=p/255.0
    out[:,:,0]=(255*t).astype(np.uint8)
    out[:,:,1]=(255*(1-np.abs(t-0.5)*2)).clip(0,255).astype(np.uint8)
    out[:,:,2]=(255*(1-t)).astype(np.uint8)
    return Image.fromarray(out)

# ---------- Header ----------
st.markdown('<div class="main-title">💧 JAL-DRISHTI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI + GIS Watershed Intelligence & Decision-Support Prototype</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("Prototype Controls")
    demo = st.toggle("Use built-in demo landscape", value=True)
    st.caption("Prototype mode: the scoring/segmentation is a transparent demonstration pipeline. Replace the heuristic stage with a trained geospatial ML model for production.")
    st.divider()
    st.markdown("**Pipeline**")
    st.write("1. Geo-coded imagery")
    st.write("2. Image preprocessing")
    st.write("3. Land-cover analysis")
    st.write("4. Terrain/drainage proxy")
    st.write("5. Priority scoring")
    st.write("6. Explainable recommendations")

if demo:
    # Create a synthetic but realistic-looking landscape for a reliable demo
    H,W=420,620
    yy,xx=np.mgrid[0:H,0:W]
    base=np.zeros((H,W,3),dtype=np.float32)
    rng=np.random.default_rng(7)
    noise=rng.normal(0,1,(H,W))
    # vegetation zones
    veg=((xx-180)**2/(170**2)+(yy-210)**2/(150**2) < 1)
    veg2=((xx-480)**2/(150**2)+(yy-125)**2/(110**2) < 1)
    # water corridor
    river=np.abs(yy-(0.30*xx+70+35*np.sin(xx/70)))<10
    # bare zones
    bare=((xx-350)**2/(120**2)+(yy-320)**2/(90**2)<1) | ((xx-90)**2/(80**2)+(yy-310)**2/(75**2)<1)
    base[:]=[178,158,112]
    base[veg]=[82,145,68]
    base[veg2]=[65,135,70]
    base[bare]=[194,151,82]
    base[river]=[58,132,205]
    # built clusters
    built=((xx-510)**2+(yy-300)**2<55**2)
    base[built]=[150,125,130]
    base += noise[:,:,None]*5
    base=np.clip(base,0,255).astype(np.uint8)
    img=Image.fromarray(base)
else:
    up=st.file_uploader("Upload a geo-coded image (JPG/PNG)", type=["jpg","jpeg","png"])
    if up is None:
        st.info("Upload an image, or turn on the built-in demo landscape.")
        st.stop()
    img=Image.open(up)

img=resize_image(img)
cls,slope,drainage,priority,zones=analyse(img)
seg=class_image(cls)
hm=heatmap(priority)

# ---------- Metrics ----------
total=zones.size
veg_pct=(cls==1).mean()*100
water_pct=(cls==2).mean()*100
bare_pct=(cls==3).mean()*100
high_pct=zones.mean()*100
mean_priority=priority.mean()*100

c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Vegetation", f"{veg_pct:.1f}%")
c2.metric("Water", f"{water_pct:.1f}%")
c3.metric("Bare soil", f"{bare_pct:.1f}%")
c4.metric("High-priority zone", f"{high_pct:.1f}%")
c5.metric("Mean priority", f"{mean_priority:.1f}/100")

st.subheader("1. Geo-coded Image → Land-cover Interpretation")
a,b=st.columns(2)
with a:
    st.image(img, caption="Input imagery", use_container_width=True)
with b:
    st.image(seg, caption="Prototype land-cover segmentation", use_container_width=True)

st.caption("Legend: green = vegetation · blue = water · tan = bare/soil · purple/grey = built/other")

st.subheader("2. GIS-style Priority Analysis")
a,b=st.columns(2)
with a:
    st.image(hm, caption="Watershed intervention priority heatmap", use_container_width=True)
with b:
    # overlay high-priority zones on original
    ov=np.asarray(img).copy()
    mask=zones
    ov[mask] = (0.55*ov[mask] + 0.45*np.array([230,45,45])).astype(np.uint8)
    st.image(Image.fromarray(ov), caption="Priority zones over imagery", use_container_width=True)

# ---------- Explanation ----------
st.subheader("3. Explainable Decision Support")
col1,col2=st.columns(2)
with col1:
    st.markdown('<div class="card"><b>Priority logic used in this prototype</b><br><br>40% bare-soil signal<br>28% terrain/slope proxy<br>18% low-vegetation signal<br>14% drainage proximity proxy<br><br><span class="small">These weights are demonstration parameters, not validated field coefficients.</span></div>', unsafe_allow_html=True)
with col2:
    recs=[]
    if bare_pct > 15: recs.append("Prioritize soil/moisture-conservation assessment in exposed/bare zones.")
    if veg_pct < 45: recs.append("Consider vegetation-restoration assessment in low-cover areas.")
    if water_pct > 2: recs.append("Review drainage/water-body connectivity and runoff pathways.")
    recs.append("Validate priority zones with field observations and domain experts before implementation.")
    st.markdown('<div class="card"><b>Generated recommendations</b><br><br>' + "<br>".join("• "+x for x in recs) + '</div>', unsafe_allow_html=True)

# ---------- Zone table ----------
st.subheader("4. Priority Zone Summary")
# Create 4x4 grid summary
rows=[]
h,w=priority.shape
for i in range(4):
    for j in range(4):
        block=priority[i*h//4:(i+1)*h//4,j*w//4:(j+1)*w//4]
        rows.append({"Zone":f"Z{i+1}{j+1}","Priority score":round(float(block.mean()*100),1),
                     "Priority level":"High" if block.mean()>=np.quantile(priority,0.75) else ("Medium" if block.mean()>=np.quantile(priority,0.45) else "Low")})
df=pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Architecture ----------
st.subheader("5. End-to-End Architecture")
st.code("""Geo-coded imagery
       ↓
Pre-processing
       ↓
AI / Computer Vision
       ↓
Land-cover features
       ↓
GIS + terrain + drainage analysis
       ↓
Priority scoring
       ↓
Explainable recommendations
       ↓
JAL-DRISHTI dashboard / report""", language="text")

# ---------- Download ----------
report = f"""JAL-DRISHTI Prototype Report
Vegetation: {veg_pct:.1f}%
Water: {water_pct:.1f}%
Bare soil: {bare_pct:.1f}%
High-priority zone: {high_pct:.1f}%
Mean priority: {mean_priority:.1f}/100

Prototype note:
This demonstration uses transparent image-processing heuristics and a priority-scoring model to demonstrate the proposed workflow. Production deployment should replace these stages with validated ML models, authoritative geospatial datasets and field verification.
"""
st.download_button("Download prototype summary", report, file_name="JAL-DRISHTI_prototype_report.txt")
st.caption("JAL-DRISHTI • SIH26015 • Prototype demonstration")
