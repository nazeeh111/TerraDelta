const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const root = require('node:path').resolve(__dirname, '..');
class NodeStub {
  constructor() { this.children = []; this.listeners = {}; this.attrs = {}; this.value = ''; this.checked = false; this.disabled = false; this.hidden = false; this.clientWidth = 100; this.style = {}; this.src = '';  this.classList = { add(){}, remove(){}, toggle(){} }; }
  append(child) { this.children.push(child); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(key, value) { this.attrs[key] = value; if (key === 'id') nodes.set(value,this); }
  getAttribute(key) { return this.attrs[key] ?? null; }
  focus() { this.focused = true; }
  getBoundingClientRect() { return {top:260,width:this.clientWidth,height:this.clientWidth,left:0,right:this.clientWidth,bottom:260+this.clientWidth}; }
  removeAttribute(key) { delete this.attrs[key]; if (key === 'href') delete this.href; }
  addEventListener(key, fn) { this.listeners[key] = fn; }
  getContext() { return { clearRect(){}, setLineDash(){}, stroke(){}, fill(){} }; }
}
const nodes = new Map();
const get = id => { if (!nodes.has(id)) nodes.set(id, new NodeStub()); return nodes.get(id); };
get('include-retracted').checked = true;
get('state-filter').value = 'confirmed_by_model';
get('start').value = '2018-01-01'; get('end').value = '2019-12-31';
const calls = [];
let sequence = 0;
let fetchMode = null;
const response = snapshot => ({ok:true, json:async () => ({type:'FeatureCollection',features:[],snapshot_time:snapshot})});
const context = {
  document:{getElementById:get, createElement:()=>new NodeStub(),createTextNode:text=>text},
  Node:NodeStub, URL, URLSearchParams, AbortController,
  window:{innerHeight:720,matchMedia:()=>({matches:true}),addEventListener(){}},
  Image:class {set src(value){this._src=value;this.onload?.();} get src(){return this._src;}},
  Path2D:class {moveTo(){} lineTo(){} closePath(){}},
  ResizeObserver:class {observe(){}},
  fetch:(url, options) => {
    calls.push({url,options});
    if (fetchMode) return fetchMode(url,options);
    return Promise.resolve(response(`2026-10-29T12:00:${String(++sequence).padStart(2,'0')}+00:00`));
  },
};
const source = fs.readFileSync(root+'/src/terradelta/web/app.js','utf8');
assert.equal(source.match(/  initialize\(\);/g).length,1);
vm.runInNewContext(source.replace('  initialize();','  globalThis.uiTest = { state, queryParameters, queryHistory, syncStateChoice, supportingImages, selectEvent, showEventList, renderStudyStatus, fitScene, renderEvents, drawOverlay };'),context);
const api = context.uiTest;
api.state.region = {region_id:'test-region',bbox:[0,0,1,1],bounds_native:[0,0,1,1],width:100,height:100,crs:'EPSG:3857',frames:[]};
const params = call => new URL(call.url,'http://localhost').searchParams;
const tick = () => new Promise(resolve=>setImmediate(resolve));
const cases = [];
function exportsMatch(expectedState, expectedRetraction) {
  for(const id of ['export-geojson','export-parquet']) {
    const p = new URL(get(id).href,'http://localhost').searchParams;
    assert.equal(p.get('state'),expectedState);
    assert.equal(p.get('include_retracted'),String(expectedRetraction));
    assert.equal(p.get('as_of'),api.state.snapshot);
  }
}
(async () => {
  const html=fs.readFileSync(root+'/src/terradelta/web/index.html','utf8');
  const select=html.match(/<select id="state-filter">(.*?)<\/select>/s)[1];
  assert.match(select,/^<option value="confirmed_by_model" selected>/);
  for(const state of ['confirmed_by_model','proposed','reviewed','retracted','all']) assert.ok(select.includes(`value="${state}"`));
  await api.queryHistory();
  assert.equal(params(calls.at(-1)).get('state'),'confirmed_by_model');
  exportsMatch('confirmed_by_model',true);
  cases.push('confirmed default and state-bound exports');

  get('state-filter').value='proposed';get('state-filter').listeners.change();await tick();
  assert.equal(params(calls.at(-1)).get('state'),'proposed');exportsMatch('proposed',true);
  get('state-filter').value='reviewed';get('state-filter').listeners.change();await tick();
  assert.equal(params(calls.at(-1)).get('state'),'reviewed');exportsMatch('reviewed',true);
  cases.push('state choices reissue history queries and exports');

  get('state-filter').value='all';get('include-retracted').checked=false;get('include-retracted').listeners.change();await tick();
  assert.equal(params(calls.at(-1)).has('state'),false);exportsMatch(null,false);
  get('state-filter').value='retracted';get('state-filter').listeners.change();await tick();
  assert.equal(get('include-retracted').checked,true);assert.equal(get('include-retracted').disabled,true);exportsMatch('retracted',true);
  get('state-filter').value='all';get('state-filter').listeners.change();await tick();
  assert.equal(get('include-retracted').checked,false);assert.equal(get('include-retracted').disabled,false);exportsMatch(null,false);
  cases.push('all excludes unchecked retractions; explicit retracted overrides and restores preference');

  let resolveOld;
  fetchMode=()=>new Promise(resolve=>{resolveOld=resolve;});
  get('state-filter').value='proposed';const old=api.queryHistory();const oldCall=calls.at(-1);
  fetchMode=null;get('state-filter').value='confirmed_by_model';await api.queryHistory();const newest=api.state.snapshot;
  assert.equal(oldCall.options.signal.aborted,true);
  resolveOld(response('2020-01-01T00:00:00+00:00'));await old;
  assert.equal(api.state.snapshot,newest);exportsMatch('confirmed_by_model',false);
  cases.push('stale responses cannot overwrite newer state/snapshot/export');

  fetchMode=()=>new Promise(resolve=>{resolveOld=resolve;});
  const pending=api.queryHistory();const pendingCall=calls.at(-1);
  get('as-of').value='invalid';await api.queryHistory();
  assert.equal(pendingCall.options.signal.aborted,true);
  resolveOld(response('2020-01-01T00:00:00+00:00'));await pending;
  assert.equal(api.state.snapshot,null);
  for(const id of ['export-geojson','export-parquet'])assert.equal(get(id).href,undefined);
  assert.equal(get('query-button').disabled,false);
  assert.match(get('query-feedback').textContent,/snapshot must be an ISO timestamp/);
  cases.push('invalid new query invalidates old pending result and exports');
  get('as-of').value='';
  const frames=Array.from({length:12},(_,i)=>({source_id:`source-${i}`,sha256:String(i).padStart(2,'0').repeat(32),acquisition_start:`2018-${String(i+1).padStart(2,'0')}-01T00:00:00Z`,acquisition_end:`2018-${String(i+1).padStart(2,'0')}-28T23:59:59Z`,valid_fraction:1,registration:{usable:true,shift_pixels:[0,0]},uri:`https://example.test/source-${i}.tif`}));
  api.state.region={...api.state.region,width:1024,height:1023,frames};
  const properties={event_id:'td-ui-event',revision:3,state:'confirmed_by_model',change_class:'building_addition',observation_interval:{lower:frames[4].acquisition_start,upper:frames[7].acquisition_end},detected_at:frames[11].acquisition_end,record_time:'2026-10-29T08:00:00Z',sources:frames.map(frame=>({...frame})),quality:{support_before_index:4,support_first_index:7,support_current_index:11},confidence:0.1,area_m2:110,model_revision:'model-sha-unit-fixture'};
  const textOf=node=>typeof node==='string'?node:(node.textContent||'')+node.children.map(textOf).join(' ');
  const descendants=node=>[node,...node.children.filter(child=>child instanceof NodeStub).flatMap(descendants)];
  let thumbs=api.supportingImages(properties.sources,properties,properties.event_id);
  assert.equal(thumbs.children.length,3);
  assert.deepEqual(thumbs.children.map(button=>Number(button.children[0].attrs.src.match(/frames\/(\d+)/)[1])),[4,7,11]);
  for(const button of thumbs.children){const p=new URL(button.children[0].attrs.src,'http://localhost').searchParams;assert.equal(p.get('detail_event'),'td-ui-event');assert.equal(p.get('detail_revision'),'3');assert.match(textOf(button),/Native-pixel detail/);}
  assert.match(textOf(thumbs.children[0]),/Before change/);assert.match(textOf(thumbs.children[1]),/First changed support/);assert.match(textOf(thumbs.children[2]),/Confirmation support/);
  thumbs.children[0].listeners.click();assert.equal(get('scene-image').src,frameUrlExpected(4));
  function frameUrlExpected(index){return `/api/regions/test-region/frames/${index}.png`;}
  assert.equal(get('scene-shell').style.width,'100px');
  cases.push('support roles select exact cited source IDs/hashes and revision crops; click restores full AOI');

  const merged={...properties,state:'proposed',quality:{...properties.quality,support_current_index:7}};
  thumbs=api.supportingImages(merged.sources,merged,merged.event_id);assert.equal(thumbs.children.length,2);assert.match(textOf(thumbs),/First changed support \/ Current proposed support/);
  const badSources=properties.sources.map((source,i)=>i===4?{...source,sha256:'f'.repeat(64)}:source);
  thumbs=api.supportingImages(badSources,properties,properties.event_id);assert.equal(thumbs.children.length,2);assert.ok(!thumbs.children.some(button=>button.children[0].attrs.src.includes('frames/4.png')));
  const generic=api.supportingImages(properties.sources,{...properties,quality:{}},properties.event_id);assert.match(textOf(generic),/Cited observation/);assert.ok(!textOf(generic).includes('Before change'));
  cases.push('duplicate support indices merge roles; mismatched bytes excluded; missing indices use generic cited roles');

  api.renderStudyStatus('<img src=x onerror=alert(1)> failed study');assert.equal(get('study-status-copy').textContent,'<img src=x onerror=alert(1)> failed study');assert.equal(get('study-status-copy').children.length,0);
  cases.push('study status remains plain text without markup execution');

  api.state.features=[{type:'Feature',id:properties.event_id,properties,geometry:{type:'Polygon',coordinates:[[[0,0],[0.001,0],[0.001,0.001],[0,0.001],[0,0]]]}}];
  api.state.snapshot='2026-10-29T08:00:00Z';
  fetchMode=()=>Promise.resolve({ok:true,json:async()=>({revisions:[{...properties,revision:3},{...properties,revision:4,state:'retracted',detected_at:'2019-01-31T23:59:59Z',record_time:'2026-10-29T08:00:01Z'}]})});
  await api.selectEvent(api.state.features[0]);
  assert.equal(get('event-list').hidden,true);assert.equal(get('event-detail').hidden,false);
  assert.equal(get('clear-selection').hidden,false);
  const detailNodes=descendants(get('event-detail'));
  assert.equal(detailNodes.filter(node=>node.attrs.href?.startsWith('https://example.test/')).length,12);
  assert.match(textOf(get('revision-content')),/Source available/);assert.match(textOf(get('revision-content')),/Recorded/);assert.match(textOf(get('revision-content')),/Returned by this query/);
  assert.match(textOf(get('event-detail')),/Returned answer · revision 3/);
  assert.equal(api.state.features[0].properties.revision,3);
  assert.ok(!get('event-detail').children.includes(null));
  get('clear-selection').listeners.click();assert.equal(get('event-list').hidden,false);assert.equal(get('event-detail').hidden,true);
  cases.push('inspector replaces list, retains all source links, separates clocks and keeps returned revision despite newer history');

  get('observation-controls').clientWidth=600;api.fitScene();assert.equal(get('scene-shell').style.width,'310px');assert.equal(get('scene-shell').style.aspectRatio,'1024 / 1023');assert.equal(get('event-overlay').width,1024);assert.equal(get('event-overlay').height,1023);
  cases.push('scene width respects viewport height with native aspect and unchanged canvas grid');
  fetchMode=null;
  get('state-filter').value='retracted';
  await api.queryHistory();
  api.state.features=Array.from({length:9550},(_,index)=>({type:'Feature',id:`retracted-${index}`,properties:{...properties,event_id:`retracted-${index}`,state:'retracted'},geometry:{type:'Polygon',coordinates:[[[0,0],[0.001,0],[0.001,0.001],[0,0.001],[0,0]]]}}));
  api.state.eventPage=0;api.state.selected=null;api.renderEvents();
  assert.equal(get('event-list').children.length,100);
  assert.equal(get('event-page-range').textContent,'1–100 of 9,550 events');
  assert.equal(get('event-page-previous').disabled,true);assert.equal(get('event-page-next').disabled,false);
  const beforePageCalls=calls.length, beforePageSnapshot=api.state.snapshot;
  const beforePageExports=['export-geojson','export-parquet'].map(id=>get(id).href);
  assert.ok(beforePageExports.every(url=>typeof url==='string' && new URL(url,'http://localhost').searchParams.get('as_of')===beforePageSnapshot));
  get('show-overlay').checked=true;api.drawOverlay();assert.equal(api.state.paths.length,9550);
  const beforePagePaths=api.state.paths;
  const ids=new Set(get('event-list').children.map(button=>button.attrs['data-event-id']));
  for(let page=1;page<96;page++){
    get('event-page-next').listeners.click();
    assert.equal(get('event-list').children.length,page===95?50:100);
    for(const button of get('event-list').children)ids.add(button.attrs['data-event-id']);
  }
  assert.equal(ids.size,9550);
  assert.equal(get('event-page-range').textContent,'9,501–9,550 of 9,550 events');
  assert.equal(get('event-page-next').disabled,true);assert.equal(get('event-page-previous').disabled,false);
  get('event-page-next').listeners.click();assert.equal(api.state.eventPage,95);
  get('event-page-previous').listeners.click();assert.equal(api.state.eventPage,94);
  assert.equal(get('event-list').scrollTop,0);assert.equal(get('event-page-range').focused,true);
  assert.equal(calls.length,beforePageCalls);assert.equal(api.state.snapshot,beforePageSnapshot);
  assert.deepEqual(['export-geojson','export-parquet'].map(id=>get(id).href),beforePageExports);
  assert.equal(api.state.paths,beforePagePaths);assert.equal(api.state.features.length,9550);
  cases.push('9,550-result pagination bounds DOM to 100, reaches every event and leaves full map, snapshot, exports and requests unchanged');

  const selectedId=get('event-list').children[7].attrs['data-event-id'];
  await get('event-list').children[7].listeners.click();
  assert.equal(get('event-pagination').hidden,true);
  get('clear-selection').listeners.click();
  assert.equal(api.state.eventPage,94);assert.equal(get('event-pagination').hidden,false);
  assert.equal(get('event-list').children[7].attrs['data-event-id'],selectedId);
  assert.equal(get('event-list').children[7].focused,true);
  cases.push('inspector Back restores prior page and selected-card keyboard focus');

  await api.queryHistory();assert.equal(api.state.eventPage,0);assert.equal(get('event-pagination').hidden,true);
  api.state.features=Array.from({length:101},(_,index)=>({properties:{...properties,event_id:`new-${index}`,state:'retracted'}}));api.renderEvents();
  get('event-page-next').listeners.click();assert.equal(api.state.eventPage,1);
  get('state-filter').value='reviewed';get('state-filter').listeners.change();await tick();
  assert.equal(api.state.eventPage,0);assert.equal(get('event-pagination').hidden,true);
  assert.equal(params(calls.at(-1)).get('state'),'reviewed');
  cases.push('new queries and state changes reset pagination and hide it for empty responses');
  console.log(JSON.stringify({passed:cases.length,cases,rendered:false,scope:'Source-level DOM/network regression harness; no actual imagery or event results'},null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
