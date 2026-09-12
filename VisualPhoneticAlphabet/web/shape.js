'use strict';
// Contour topology: MediaPipe FACEMESH_LIPS, Apache-2.0.
// https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
const VPA_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
const VPA_INNER = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95];
const VPA_SHAPE_FIELDS = {
 opening_area:'Opening area (eye-distance²)',
 opening_area_ratio:'Opening area / width²',
 side_gap_61:'Side gap toward point 61',
 side_gap_291:'Side gap toward point 291',
 gap_asymmetry:'Side-gap asymmetry / width',
 upper_lip_thickness:'Upper lip thickness (2D)',
 lower_lip_thickness:'Lower lip thickness (2D)',
};
function shapeMeasurements(frame) {
 const values=Object.fromEntries(Object.keys(VPA_SHAPE_FIELDS).map(k=>[k,null]));
 const p=frame.landmarks;
 if(frame.quality.status!=='observed'||!p)return values;
 const valid=id=>Array.isArray(p[id])&&p[id].length===2&&p[id].every(Number.isFinite);
 const distance=(a,b)=>valid(a)&&valid(b)?Math.hypot(p[a][0]-p[b][0],p[a][1]-p[b][1]):null;
 const width=distance(61,291);
 values.side_gap_61=distance(81,178);
 values.side_gap_291=distance(311,402);
 values.upper_lip_thickness=distance(0,13);
 values.lower_lip_thickness=distance(14,17);
 if(VPA_INNER.every(valid)){
  let twiceArea=0;
  for(let i=0;i<VPA_INNER.length;i++){
   const a=p[VPA_INNER[i]],b=p[VPA_INNER[(i+1)%VPA_INNER.length]];
   twiceArea+=a[0]*b[1]-b[0]*a[1];
  }
  values.opening_area=Math.abs(twiceArea)/2;
  if(width>1e-6)values.opening_area_ratio=values.opening_area/(width*width);
 }
 if(width>1e-6 && values.side_gap_61!==null && values.side_gap_291!==null)
  values.gap_asymmetry=Math.abs(values.side_gap_61-values.side_gap_291)/width;
 return values;
}
