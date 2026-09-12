'use strict';
const $ = id => document.getElementById(id);
const fields = {inner_aperture:'Inner aperture',outer_aperture:'Outer aperture',mouth_width:'Mouth width',width_to_height:'Width / height',rounding:'Rounding proxy',bilabial_contact:'Closure proxy',opening_velocity:'Opening velocity',landmark_speed:'Landmark speed'};
const baseFields = {...fields};
Object.assign(fields,VPA_SHAPE_FIELDS);
const measurementCache=new WeakMap();
function measurements(frame){if(!measurementCache.has(frame))measurementCache.set(frame,{...frame.features,...shapeMeasurements(frame)});return measurementCache.get(frame);}
const tokens = {'BCL':'Bilabial closure','BCL-REL':'Closure release','RND':'Rounding','OPEN':'Opening','UNK-VIS':'Missing evidence'};
let records=[],record,index=0,playing=false,startTime=0,startFrame=0,geometryBounds;
const video = $('source-video');
let videoURL=null, videoCallback=null, videoCallbackKind=null;
const finite = value => typeof value === 'number' && Number.isFinite(value);
const svg = (tag,attrs,parent) => {const node=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));parent.append(node);return node;};
for(const [key,label] of Object.entries(fields)) $('feature').add(new Option(label,key));
function validate(r) {
 if(r.schema!=='vpa-0.1'||!Array.isArray(r.frames)||!r.frames.length||!finite(r.source_fps)||r.source_fps<=0||!Array.isArray(r.gestures)) throw Error('Expected a vpa-0.1 clip with frames, gestures, and a positive frame rate.');
 r.frames.forEach((f,i)=>{if(!finite(f.time_ms)||f.time_ms<0||(i&&f.time_ms<=r.frames[i-1].time_ms)||!f.features||!f.quality||!['observed','missing'].includes(f.quality.status)) throw Error('Frames must have increasing timestamps, features, and an observation status.');for(const key of Object.keys(baseFields)) if(f.features[key]!==null&&!finite(f.features[key])) throw Error('Invalid measurement: '+key);if(f.landmarks!=null&&Object.values(f.landmarks).some(p=>!Array.isArray(p)||p.length!==2||!p.every(finite))) throw Error('Landmarks must contain finite x/y pairs.');});
 r.gestures.forEach(g=>{if(!(g.token in tokens)||!finite(g.start_ms)||!finite(g.end_ms)||g.start_ms<0||g.end_ms<g.start_ms) throw Error('Invalid gesture interval.');});
 return r;
}
function stop(){cancelVideoTick();video.pause();playing=false;$('play').textContent='Play';}
function load(items){items.forEach(validate);stop();records=items;$('records').replaceChildren();items.forEach((r,i)=>$('records').add(new Option(r.clip_id||`Clip ${i+1}`,i)));select(0);$('error').textContent='';}
function select(n){stop();record=records[n];index=0;geometryBounds=[Infinity,-Infinity,Infinity,-Infinity];for(const frame of record.frames)for(const p of Object.values(frame.landmarks||{})){geometryBounds[0]=Math.min(geometryBounds[0],p[0]);geometryBounds[1]=Math.max(geometryBounds[1],p[0]);geometryBounds[2]=Math.min(geometryBounds[2],p[1]);geometryBounds[3]=Math.max(geometryBounds[3],p[1]);}$('scrub').max=record.frames.length-1;$('clip').textContent=record.clip_id;$('summary').textContent=`${record.frames.length} frames · ${record.source_fps} FPS · ${Math.round(record.frames.filter(f=>f.quality.status==='observed').length/record.frames.length*100)}% observed · ${record.gestures.length} gesture intervals`;setVideo();drawTimeline();render();}
function bounds(){return [record.frames[0].time_ms,record.frames.at(-1).time_ms+1000/record.source_fps];}
function drawTimeline(){const [a,b]=bounds();$('timeline').replaceChildren();for(const [token,meaning] of Object.entries(tokens)){const label=document.createElement('span');label.textContent=token;label.title=meaning;$('timeline').append(label);const track=document.createElement('div');track.className='track';for(const g of record.gestures.filter(g=>g.token===token)){const left=Math.max(a,g.start_ms),right=Math.min(b,g.end_ms);if(right<=left)continue;const el=document.createElement('button');el.className='event';el.style.left=`${(left-a)/(b-a)*100}%`;el.style.width=`${(right-left)/(b-a)*100}%`;el.title=`${meaning}: ${g.start_ms.toFixed(0)}–${g.end_ms.toFixed(0)} ms`;el.setAttribute('aria-label',el.title);el.onclick=()=>{stop();index=record.frames.findIndex(f=>f.time_ms>=g.start_ms);if(index<0)index=record.frames.length-1;render();};track.append(el);}const cursor=document.createElement('span');cursor.className='cursor';track.append(cursor);$('timeline').append(track);}}
function render(syncVideo=true, mediaTime=null){const f=record.frames[index];$('scrub').value=index;$('time').textContent=`${f.time_ms.toFixed(0)} ms`;$('status').textContent=`Frame ${index+1} / ${record.frames.length} · ${f.quality.status==='observed'?'Observed':'Missing visual evidence'}`;$('measurements').replaceChildren();for(const [key,label] of Object.entries(fields)){const row=document.createElement('tr');const name=document.createElement('th');name.scope='row';name.textContent=label;const val=document.createElement('td');val.textContent=finite(measurements(f)[key])?measurements(f)[key].toFixed(3):'Unknown';row.append(name,val);$('measurements').append(row);}
 const mouth=$('mouth');mouth.replaceChildren();if(f.landmarks&&Object.keys(f.landmarks).length){const [minX,maxX,minY,maxY]=geometryBounds;const s=Math.min(370/Math.max(maxX-minX,.1),190/Math.max(maxY-minY,.1));const point=p=>[230+(p[0]-(minX+maxX)/2)*s,135+(p[1]-(minY+maxY)/2)*s];for(const ids of [[...VPA_OUTER,61],[...VPA_INNER,78]]){if(ids.every(id=>f.landmarks[id]))svg('polyline',{points:ids.map(id=>point(f.landmarks[id]).join(',')).join(' '),fill:'none',stroke:'#236b53','stroke-width':2},mouth);}for(const [id,p] of Object.entries(f.landmarks)){const [x,y]=point(p);svg('circle',{cx:x,cy:y,r:3,fill:'#233730'},mouth);svg('text',{x:x+7,y:y-7,fill:'#506159','font-size':12},mouth).textContent=id;}}else svg('text',{x:230,y:135,'text-anchor':'middle',fill:'#506159'},mouth).textContent='No landmarks available';
 const [a,b]=bounds();document.querySelectorAll('.cursor').forEach(el=>el.style.left=`${(f.time_ms-a)/(b-a)*100}%`);const active=record.gestures.filter(g=>f.time_ms>=g.start_ms&&f.time_ms<g.end_ms);$('active').textContent=active.length?'Active: '+active.map(g=>`${g.token} · ${tokens[g.token]}`).join(' / '):'No active gesture at this frame.';drawChart();if(syncVideo && video.readyState>=1) video.currentTime=f.time_ms/1000;drawVideoPoints(syncVideo?f.time_ms:mediaTime);}
function drawChart(){const key=$('feature').value,chart=$('chart');chart.replaceChildren();const vals=record.frames.map(f=>measurements(f)[key]).filter(finite);if(!vals.length){$('scale').textContent='No observed values for this measurement.';return;}let lo=Infinity,hi=-Infinity;for(const v of vals){lo=Math.min(lo,v);hi=Math.max(hi,v);}const [a,b]=bounds();let d='';let pen=false;for(const f of record.frames){const v=measurements(f)[key];if(!finite(v)){pen=false;continue;}d+=`${pen?'L':'M'}${20+(f.time_ms-a)/(b-a)*960},${115-(v-lo)/(hi-lo||1)*95} `;pen=true;}svg('path',{d,fill:'none',stroke:'#236b53','stroke-width':2},chart);svg('line',{x1:20+(record.frames[index].time_ms-a)/(b-a)*960,x2:20+(record.frames[index].time_ms-a)/(b-a)*960,y1:10,y2:125,stroke:'#233730'},chart);$('scale').textContent=`${a.toFixed(0)}–${b.toFixed(0)} ms · Range ${lo.toFixed(3)}–${hi.toFixed(3)} · Lengths: eye-distance units; area: eye-distance²; velocities: units/second; ratios: unitless. Gaps indicate unknown values.`;}
$('scrub').oninput=()=>{stop();index=Number($('scrub').value);render();};$('feature').onchange=drawChart;$('records').onchange=()=>select(Number($('records').value));$('demo').onclick=()=>load([window.VPA_SAMPLE]);
$('play').onclick=()=>{
 if(playing || !video.paused){stop();return;}
 if(index===record.frames.length-1)index=0;
 render();
 if(video.readyState>=1 && !video.error){video.play().catch(()=>{$('video-status').textContent='Playback could not start. Use the video controls or choose a browser-compatible video.';});return;}
 playing=true;$('play').textContent='Pause';startTime=performance.now();startFrame=record.frames[index].time_ms;requestAnimationFrame(tick);
};
function tick(now){if(!playing)return;const time=startFrame+(now-startTime)*Number($('speed').value);while(index<record.frames.length-1&&record.frames[index+1].time_ms<=time)index++;render(false);if(time>=bounds()[1])stop();else requestAnimationFrame(tick);}
function setVideo(){
 if(videoURL){URL.revokeObjectURL(videoURL);videoURL=null;}
 video.removeAttribute('src');video.hidden=true;video.load();
 if(record.clip_id===window.VPA_SAMPLE.clip_id){
  video.src='sample.mp4';video.hidden=false;
  $('video-status').textContent='Original AVSpeech sample · measurements use the manual speaker region.';
 }else $('video-status').textContent='Choose the source video for this clip to view it alongside the data.';
}
function syncFromVideo(mediaTime=video.currentTime){
 if(video.readyState<1)return;
 const time=mediaTime*1000;
 let lo=0,hi=record.frames.length-1;
 while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(record.frames[mid].time_ms<=time)lo=mid;else hi=mid-1;}
 index=lo;render(false, time);
}
function cancelVideoTick(){
 if(videoCallback!==null){
  if(videoCallbackKind==='video')video.cancelVideoFrameCallback(videoCallback);
  else cancelAnimationFrame(videoCallback);
 }
 videoCallback=null;
}
function scheduleVideoTick(){
 cancelVideoTick();
 if(video.paused||video.ended)return;
 if(video.requestVideoFrameCallback){
  videoCallbackKind='video';
  videoCallback=video.requestVideoFrameCallback((now,metadata)=>{videoCallback=null;syncFromVideo(metadata.mediaTime);scheduleVideoTick();});
 }else{
  videoCallbackKind='animation';
  videoCallback=requestAnimationFrame(()=>{videoCallback=null;syncFromVideo();scheduleVideoTick();});
 }
}
video.onplay=()=>{playing=false;$('play').textContent='Pause';scheduleVideoTick();};
video.onpause=()=>{cancelVideoTick();$('play').textContent='Play';syncFromVideo();};
video.onseeked=()=>syncFromVideo();
video.onloadedmetadata=()=>{video.currentTime=record.frames[index].time_ms/1000;drawVideoPoints();};
video.onerror=()=>{$('video-status').textContent='Could not load this video. Choose the matching source file in a format your browser supports.';};
$('video-file').onchange=()=>{
 const file=$('video-file').files[0];if(!file)return;
 stop();if(videoURL)URL.revokeObjectURL(videoURL);
 videoURL=URL.createObjectURL(file);video.src=videoURL;video.hidden=false;
 $('video-status').textContent=file.name;$('video-file').value='';
};
$('file').onchange=async()=>{const file=$('file').files[0];if(!file)return;try{if(file.size>20*1024*1024)throw Error('Choose a file smaller than 20 MB.');const raw=await file.text();let parsed;try{parsed=JSON.parse(raw);}catch{parsed=raw.split(/\r?\n/).filter(s=>s.trim()).map(line=>JSON.parse(line));}const items=Array.isArray(parsed)?parsed:[parsed];if(!items.length)throw Error('The file contains no clips.');load(items);}catch(e){$('error').textContent=`Could not load file. ${e.message} Choose a VPA JSON or JSONL export.`;}finally{$('file').value='';}};
$('show-overlay').onchange=()=>{$('frame-overlay').hidden=!$('show-overlay').checked;};
function drawVideoPoints(mediaTime=null){
 const layer=$('video-points');layer.replaceChildren();
 if(!$('show-points').checked){$('tracking-status').textContent='Mouth points hidden.';return;}
 const f=record.frames[index],size=f.quality.source_size,points=f.quality.source_landmarks;
 const time=finite(mediaTime)?mediaTime:(video.readyState>=1?video.currentTime*1000:f.time_ms);
 if(time<bounds()[0] || time>=bounds()[1]){$('tracking-status').textContent='No tracking data at this video time.';return;}
 if(f.quality.status!=='observed'){$('tracking-status').textContent='No mouth detection in this frame.';return;}
 if(!Array.isArray(size)||size.length!==2||!size.every(v=>finite(v)&&v>0)||!points){$('tracking-status').textContent='No source-position landmarks for this frame. Older exports need to be re-extracted.';return;}
 const width=video.clientWidth,height=video.clientHeight;
 const scale=Math.min(width/size[0],height/size[1]);
 layer.style.width=`${size[0]*scale}px`;layer.style.height=`${size[1]*scale}px`;
 layer.style.left=`${video.offsetLeft+(width-size[0]*scale)/2}px`;layer.style.top=`${video.offsetTop+(height-size[1]*scale)/2}px`;
 layer.setAttribute('viewBox',`0 0 ${size[0]} ${size[1]}`);
 let pointCount=0;
 for(const [ids,color] of [[VPA_OUTER,'#70ffab'],[VPA_INNER,'#ffcf70']]){
 if(ids.every(id=>Array.isArray(points[id])&&points[id].length===2&&points[id].every(finite)))svg('polygon',{points:ids.map(id=>points[id].join(',')).join(' '),fill:'none',stroke:color,'stroke-width':1,'vector-effect':'non-scaling-stroke'},layer);
 for(const id of ids){const p=points[id];if(!Array.isArray(p)||p.length!==2||!p.every(finite))continue;
 pointCount++;
 svg('circle',{cx:p[0],cy:p[1],r:Number($('point-size').value)/Math.max(scale,.1),fill:color,stroke:'#102e20','stroke-width':1,'vector-effect':'non-scaling-stroke'},layer);
 }
 }
 $('tracking-status').textContent=`${pointCount} lip points · green: outer lip · amber: inner opening · source video coordinates`;
}
$('show-points').onchange=()=>drawVideoPoints();
$('point-size').oninput=()=>drawVideoPoints();
$('speed').onchange=()=>{video.playbackRate=Number($('speed').value);};
for(const [id,delta] of [['previous-frame',-1],['next-frame',1]])$(id).onclick=()=>{stop();index=Math.max(0,Math.min(record.frames.length-1,index+delta));render();};
new ResizeObserver(()=>{if(record)drawVideoPoints();}).observe(video);
load([window.VPA_SAMPLE]);
