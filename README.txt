比奇堡居民手机壁纸

定位
  海绵宝宝（SpongeBob SquarePants）角色手机壁纸，竖屏 1440x3200。
  个人壁纸用途；Qwen-Image-2.1 产图仅限非商业（不得上店铺）。

技术路线（按可行性排序）
  1  Qwen-Image-2.1 文生图直出 — 已验证成功，全员合影版
     脚本 scripts\comfyui\qwen-image-2.1\run-t2i-portrait.cjs
     种子 20261103，prompt 见 prompts\group-all.txt
  2  Illustrious XL 角色 LoRA + 单位合成 — 自训海绵宝宝 LoRA（r5 轮），
     多角色同框困难（属性互串），单人可出；合成需修 alpha 边缘
  3  实拍底图（Krea）→ img2img 动画化（denoise 0.5）→ 单位合成 —
     背景质感最好，但角色合成 halo 未解决

角色清单
  海绵宝宝 SpongeBob、派大星 Patrick、章鱼哥 Squidward、
  蟹老板 Mr. Krabs、珊迪 Sandy、小蜗 Gary、痞老板 Plankton

素材与训练
  assets\<角色>\ 各角色参考图（百度/Bing/Fandom 抓取）
  训练在 G:\workspace\trainer\（sd-scripts + Illustrious 底模）
  LoRA 产出 trainer\output-rX\sbp-rX.safetensors

尺寸规格
  标准竖屏 1440x3200（32 倍数 1440x3136 为 Qwen-Image 实际出图）
  低分辨率预览 960x2080
  放大用 Remacri 到目标尺寸

许可边界
  Qwen-Image-2.1：仅限非商业
  Illustrious XL：CreativeML OpenRAIL-M
  海绵宝宝角色：Nielsen/Paramount 版权，个人使用不侵权

脚本索引（G:\workspace\scripts\wallpaper-spongebob\）
  run-t2i-portrait.cjs      Qwen-Image 竖版生图（在 scripts\comfyui\qwen-image-2.1\）
  fetch-*.cjs               素材抓取（百度/Bing/Fandom/PNGSite）
  curate.py                 抠图去重清洗
  build-dataset.py          训练集构建（含逐图标注）
  train-ds.ps1              LoRA 训练入口
  compose-*.py/.cjs         角色 alpha + 背景合成
  contact-sheet.py          对比总览图生成
