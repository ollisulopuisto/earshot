import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source = readFileSync(new URL('../src/earshot/listen.py', import.meta.url), 'utf8');
const end = source.indexOf('</script>');
const registryStart = source.indexOf('const LISTEN_SHORTCUTS');
const start = registryStart >= 0 ? registryStart : source.indexOf('document.addEventListener("keydown"');
let handler, plays = 0, selected = [], seeks = [];
const view = {toggle:()=>plays++, visibleOrder:()=>[2,0,1], select:i=>selected.push(i), position:()=>10, seek:t=>seeks.push(t)};
const doc = {activeElement:null, querySelectorAll:()=>[], addEventListener:(_,fn)=>{handler=fn;}};
vm.runInNewContext(source.slice(start,end), {document:doc,state:{active:null},views:[view]});
const event = (key, code=key, extra={}) => ({key,code,target:{},preventDefault(){this.defaultPrevented=true;},...extra});
handler(event(' ','Space',{altKey:true}));
assert.equal(plays,0,'Option-Space must not play');
handler(event(' ','Space')); assert.equal(plays,1);
handler(event('1')); assert.deepEqual(selected,[2]);
handler(event('ArrowLeft'));handler(event('ArrowRight'));assert.deepEqual(seeks,[8,12]);
for(const [key,code] of [[' ','Space'],['2','Digit2'],['ArrowLeft','ArrowLeft'],['ArrowRight','ArrowRight']]) {
 for(const extra of [{metaKey:true},{ctrlKey:true},{altKey:true},{shiftKey:true},{isComposing:true},{keyCode:229},{defaultPrevented:true},{target:{isContentEditable:true}},{target:{closest:()=>({})}},{composedPath:()=>[{isContentEditable:true}]}]) {
  const before=[plays,selected.length,seeks.length];handler(event(key,code,extra));assert.deepEqual([plays,selected.length,seeks.length],before);
 }
}
doc.activeElement={isContentEditable:true};handler(event(' ','Space'));assert.equal(plays,1);
doc.activeElement=null;doc.querySelectorAll=()=>[{getClientRects:()=>[{}]}];handler(event(' ','Space'));assert.equal(plays,1);
doc.querySelectorAll=()=>[];handler(event('9'));assert.deepEqual(selected,[2]);
console.log('actual listening handler: bare keys, modifiers, editing, controls, dialog, IME, bounds pass');
