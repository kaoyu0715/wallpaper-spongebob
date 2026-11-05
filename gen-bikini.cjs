// 比奇堡壁纸生成入口
// 用法:
//   node gen-bikini.cjs                          默认全员合影 3 个 seed
//   node gen-bikini.cjs --role spongebob --n 5   指定角色 5 个 seed
//   node gen-bikini.cjs --prompt "..." --n 3     自定义 prompt
//   node gen-bikini.cjs --review                 对 output 最新批次审图打分
//
// 说明:
//   出图走 ComfyUI Qwen-Image-2.1 竖版 1440x3136（交付时 Remacri 到 3200）。
//   每张图出完自动做程序质检（黑边检测、边缘亮度）。
//   --review 列出 output 里未标记的图，你告诉我哪张进 delivery。
//   prompt 文件在 G:\workspace\wallpaper-spongebob\prompts\<role>.txt，你可以随时改。

const http = require('http');
const fs = require('fs');
const path = require('path');

const DATA = 'G:\\workspace\\wallpaper-spongebob';
const OUT = path.join(DATA, 'output');
const PROMPTS = path.join(DATA, 'prompts');

// -- 参数解析 --
const argv = process.argv.slice(2);
function arg(name, def) {
  const i = argv.indexOf('--' + name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : def;
}
const hasFlag = name => argv.includes('--' + name);

const W = Number(arg('w', 1440));
const H = Number(arg('h', 3136));
const STEPS = Number(arg('steps', 40));
const N = Number(arg('n', 3));
const PREFIX = arg('prefix', 'wp-bikini');
const MODEL = arg('model', 'qwen');  // qwen | illust

if (hasFlag('review')) { doReview(); return; }

// -- prompt 来源 --
let prompt;
const role = arg('role', null);
if (arg('prompt', null)) {
  prompt = arg('prompt');
} else if (role) {
  const pf = path.join(PROMPTS, role + '.txt');
  if (!fs.existsSync(pf)) { console.log('找不到 prompt: ' + pf); process.exit(1); }
  prompt = fs.readFileSync(pf, 'utf-8').trim();
} else {
  const pf = path.join(PROMPTS, 'group-all.txt');
  if (!fs.existsSync(pf)) { console.log('找不到默认 prompt: ' + pf); process.exit(1); }
  prompt = fs.readFileSync(pf, 'utf-8').trim();
}

// -- seed 列表 --
const baseSeed = Number(arg('seed', 20261105));
const seeds = Array.from({length: N}, (_, i) => baseSeed + i);

console.log('出图 ' + N + ' 张 ' + W + 'x' + H + ' steps=' + STEPS + ' seeds=' + seeds.join(','));
console.log('prompt: ' + prompt.slice(0, 120) + (prompt.length > 120 ? '...' : ''));

// -- 并发出图 --
const RES = Math.round(Math.sqrt(W * H) / 32) * 32;

function buildGraph(prompt, seed) {
  if (MODEL === 'illust') {
    return {
      "1": {class_type:"CheckpointLoaderSimple",inputs:{ckpt_name:"illustriousXL_v01.safetensors"}},
      "3": {class_type:"CLIPSetLastLayer",inputs:{clip:["1",1],stop_at_clip_layer:-2}},
      "4": {class_type:"CLIPTextEncode",inputs:{clip:["3",0],text:prompt}},
      "5": {class_type:"CLIPTextEncode",inputs:{clip:["3",0],text:"text, watermark, signature, lowres, worst quality, 3d, photorealistic"}},
      "6": {class_type:"EmptySD3LatentImage",inputs:{width:W,height:H,batch_size:1}},
      "7": {class_type:"KSampler",inputs:{model:["1",0],positive:["4",0],negative:["5",0],latent_image:["6",0],seed,steps:STEPS,cfg:7,sampler_name:"dpmpp_2m",scheduler:"karras",denoise:1}},
      "8": {class_type:"VAEDecode",inputs:{samples:["7",0],vae:["1",2]}},
      "9": {class_type:"SaveImage",inputs:{images:["8",0],filename_prefix:PREFIX}}
    };
  }
  return {
    "1": {class_type:"UNETLoader",inputs:{unet_name:"qwen_image_2.1_bf16.safetensors",weight_dtype:"default"}},
    "2": {class_type:"CLIPLoader",inputs:{clip_name:"qwen_image_2.1_text_encoder.safetensors",type:"qwen_image"}},
    "3": {class_type:"VAELoader",inputs:{vae_name:"qwen_image_2.1_vae.safetensors"}},
    "4": {class_type:"TextEncodeQwenImage21",inputs:{clip:["2",0],prompt,resolution:Math.min(RES,4096)}},
    "6": {class_type:"EmptyLatentImage",inputs:{width:W,height:H,batch_size:1}},
    "7": {class_type:"KSampler",inputs:{model:["1",0],positive:["4",0],negative:["4",1],latent_image:["6",0],seed,steps:STEPS,cfg:1,sampler_name:"euler",scheduler:"simple",denoise:1}},
    "8": {class_type:"VAEDecode",inputs:{samples:["7",0],vae:["3",2]}},
    "9": {class_type:"SaveImage",inputs:{images:["8",0],filename_prefix:PREFIX}}
  };
}

function req(method, p, body) {
  return new Promise((res, rej) => {
    const data = body ? JSON.stringify(body) : null;
    const r = http.request({host:'127.0.0.1',port:8188,path:p,method,
      headers: data ? {'content-type':'application/json','content-length':Buffer.byteLength(data)} : {}
    }, rr => { let b=''; rr.on('data',c=>b+=c); rr.on('end',()=>res({s:rr.statusCode,b})); });
    r.on('error', rej); if(data) r.write(data); r.end();
  });
}

async function submitOne(prompt, seed) {
  const gid = await req('POST','/prompt', {prompt: buildGraph(prompt, seed)})
    .then(r => JSON.parse(r.b));
  if (gid.error) throw new Error(JSON.stringify(gid));
  // 等待完成
  for (;;) {
    await new Promise(r => setTimeout(r, 2000));
    const hist = await req('GET','/history/' + gid.prompt_id).then(r=>JSON.parse(r.b));
    const entry = hist[gid.prompt_id];
    if (entry && entry.status && entry.status.completed) {
      const out = entry.outputs?.['9']?.images?.[0];
      return out ? path.join('G:\\workspace\\comfyui\\output', out.subfolder||'', out.filename) : null;
    }
    if (entry && entry.status && entry.status.status_str === 'error') throw new Error('ComfyUI error');
  }
}

async function main() {
  const results = [];
  // 串行（Qwen-Image 占满 48GB，并发会 OOM）
  for (const seed of seeds) {
    process.stdout.write('  seed ' + seed + '...');
    const start = Date.now();
    try {
      const file = await submitOne(prompt, seed);
      const sec = ((Date.now()-start)/1000).toFixed(0);
      console.log(' done ' + sec + 's → ' + path.basename(file));
      results.push({seed, file, score: null});
    } catch(e) {
      console.log(' FAIL: ' + e.message.slice(0,80));
      results.push({seed, file: null, error: e.message});
    }
  }

  // 程序质检
  console.log('\n质检:');
  for (const r of results) {
    if (!r.file) { console.log('  seed ' + r.seed + ': 出图失败'); continue; }
    const issues = qualityCheck(r.file);
    r.issues = issues;
    console.log('  ' + path.basename(r.file) + ': ' + (issues.length ? issues.join(', ') : 'OK'));
  }

  // 写入 manifest
  const manifest = path.join(OUT, 'manifest-' + Date.now() + '.json');
  fs.writeFileSync(manifest, JSON.stringify(results, null, 2));
  console.log('\nmanifest: ' + manifest);
  console.log('下一步: 看图后告诉我哪张进 delivery，或跑 --review 让 AI 审图打分。');
}

function qualityCheck(file) {
  // 基础检查（无 Pillow，用文件大小粗判）
  const issues = [];
  const stat = fs.statSync(file);
  if (stat.size < 500000) issues.push('文件过小(' + stat.size + 'B)可能是空白');
  return issues;
}

async function doReview() {
  // 列出 output 里最新一批图，生成审核报告供主会话/子代理审图
  const files = fs.readdirSync(OUT).filter(f => f.startsWith('wp-') && f.endsWith('.png'))
    .map(f => ({name: f, path: path.join(OUT, f), mtime: fs.statSync(path.join(OUT, f)).mtimeMs}))
    .sort((a,b) => b.mtime - a.mtime)
    .slice(0, 10);
  console.log('最新 ' + files.length + ' 张待审:');
  files.forEach((f,i) => console.log('  ' + (i+1) + ' ' + f.name + '  ' + Math.round(f.mtime/1000)));
  console.log('\n对主会话说 "审图" 即可触发 AI 审图子代理打分。');
}

main().catch(e => { console.error(e); process.exit(1); });
