# 壁纸生产配方（八代固化版）

> 来源：五～七代「官方图案背景线」收敛结论（PR #12 / #14 / #16，issue #11/#13/#15）。
> 本文件是**可复现交付配方**：照此执行即可稳定产出合格壁纸。八代批量生产（`delivery/batch/`）全部按本配方执行。

合格硬标准（每张必须全满足，缺一判不合格）：

- **C1** 官方 2D 画风，贴合样张 `review/reference/target-ref.jpg`
- **C2** ≤3 个角色，动作与位置互相关联、围成有叙事的小场景（不是一字并列）
- **C3** 所有角色肢体完整无缺失（海绵宝宝双臂双手双腿、手有手指；各角色尺寸协调、无放大过度）
- **C4** 描边干净无偏色、无怪道具、无文字、无畸变，竖版 1440x3200

---

## 0. 管线总览（为什么是这条线）

```
① 逐角色单元出图（illustriousXL + 角色专属 LoRA，单人白底）
② RMBG-2.0 抠图 + alpha 收紧          → 无白色 halo
③ 程序绘制官方图案背景（bg-flat.py）    → 样式 100% 可控、无 AI 伪影
④ 合成（compose-round8.py）：定向回色/清斑 → 贴入 → 泡泡带串视线
⑤ 按 C1–C4 逐张审（关键区域 100% 放大复核）
```

要点：**风格与构图靠程序保证，角色质量靠“单人小场景 + 种子选优”保证。**
多角色直接同框文生图在五代前已被否决（属性互串：粉头、绿箭头、海绵块手——七代复验仍复现，见 PR #16），
因此每个角色单独生成、单独审、再程序合成；画面里唯一“画”出来的是角色单元，背景与互动道具全部程序绘制。

---

## 1. 角色单元出图（工作流 + 采样参数）

ComfyUI 图结构（同 `gen-round8.py`，亦即 gen-round7.py 的模板化）：

| 节点 | 参数 | 为什么 |
|---|---|---|
| CheckpointLoaderSimple | `illustriousXL_v01.safetensors` | 官方 2D 卡通风的底模；Qwen 线（写实、全员合影）已停用于本线 |
| CLIPSetLastLayer | `stop_at_clip_layer = -2` | 掉最后一级，解锁 Illustrious 系风格词表达（sd-scripts 同款约定） |
| LoraLoader | 角色 LoRA，见 §2 | 每角色专属 LoRA，杜绝跨角色属性互串 |
| CLIPTextEncode(正) | 模板见 §1.1 | |
| CLIPTextEncode(负) | 模板见 §1.2 | 肢体缺陷负面清单是 C3 的一层保险 |
| EmptySD3LatentImage | **1024 x 1536** | 出图分辨率必须 ≥ 贴入尺寸（见 §4 不放大规则）。六代海绵宝宝按 832 出图、2.1x 放大贴入 → 发软+肢体崩；七代提到 1024x1536 后 0.62x 缩小贴入，全分辨率锐利 |
| KSampler | **steps=30, sampler=dpmpp_2m, scheduler=karras, cfg=5.5, denoise=1.0** | 30 步足够线条稳定；**cfg 5.5 是七代实测胜出值**——cfg 7 持续出现三只鞋/人群背景/伪影，5.5 一次命中（PR #16 manifest） |

每角色每姿态出 3–5 个 seed，逐张过 §5 单元审计后**只选 1 个干净种子**进入合成。选优不省：
七代海绵宝宝试了 22 个 seed 才命中；跳过审计直接批量贴入会把废件带进成品。

### 1.1 正面 prompt 模板

```
<角色触发词>, <角色体貌特征词>, <姿态短语>, <方向短语>,
full body with margin around, nothing cropped,
flat 2d cartoon, tv cartoon style, clean lineart, simple shading, solo, simple white background
```

- **姿态短语（海绵宝宝，七代胜出模板）**：`standing on the ground, both feet on the ground, exactly two shoes, arms relaxed slightly away from body, open hands with four spread fingers, happy smile`
  —— 双臂离身 + 手掌张开 + 四指，直接对症六代“团块拳”；“exactly two shoes”对症三只鞋伪影。
- **方向短语**：`looking to the left|right` —— 视线方向必须朝向对手角色，这是 C2 叙事关联的锚（视线链）。
- **`solo` + `simple white background`**：单人、纯色底 → 抠图干净、不带入背景杂物。
- 海绵宝宝特征词（sbpilot 训练 caption 原词，照抄）：`yellow porous sponge, square body, big blue eyes, buck teeth, white shirt, red tie, brown shorts, red striped socks, black shoes`。

### 1.2 负面 prompt 模板（固定，全角色通用）

```
text, watermark, signature, lowres, worst quality, 3d, photorealistic, realistic,
outline, greyscale, monochrome, lineart, extra eyes, extra characters,
two characters, multiple views, duplicate character, flower, plants, shadow,
missing arm, missing hands, missing leg, missing fingers, extra finger, extra arms,
extra hands, extra legs, fused fingers, mitten hands, deformed hands, blob hands,
malformed hands, clenched fist, hands hidden, arms behind back, arms crossed,
hands in pockets, holding object, cropped, cut off, out of frame, partial body
```

外加角色特异补充：海绵宝宝 `three shoes, floating shoe, detached shoe, third shoe`；
蟹老板 `money, dollar sign, coins, money bag`（其 LoRA 训练集带“抱钱云”先验，防怪道具）；
小蜗 `three eye stalks, missing eye stalk`；派大星 `green shorts, clothes pattern`（防 LoRA 把短裤画花）。
`holding object` 必带：五代泡泡棒被画成话筒/冰棒怪道具后，全线改为空手手势 + 程序绘制泡泡。

## 2. 各角色 LoRA 与权重

| 角色 | LoRA | 权重(model/clip) | 触发词 | 备注（为什么用/注意什么） |
|---|---|---|---|---|
| 海绵宝宝 | `sbpilot.safetensors` | **0.85** | `spongebob squarepants` | 七代胜出：单角色训练、caption 即 §1.1 特征词，无互串；多角色版 sbp-r7 在新姿态下持续互串（粉头/绿箭头/海绵块手）。权重 0.85 压蓝色袖泄漏。**已知轻微漂移**：手指偏细长、头部外缘一圈浅绿描边——手机尺寸不显眼，七代人工验收通过 |
| 派大星 | `Patrick_il.safetensors` | 1.0 | `p4tr1ckst4r` + `official style` | 出图**必带** §4 三件套清理（红描边回正 + 短裤花斑/蓝彩条回填），数据伪影固定、清理参数固定 |
| 章鱼哥 | `squidward-v3.safetensors` | 1.0 | `squidward tentacles` | 五~七代零漂移 |
| 珊迪 | `Sandy_il.safetensors` | 1.0 | `sandycheeks` | 训练集带 `pink flower` 高频 tag，负面里保留 `flower, plants` 防花形漂浮物 |
| 蟹老板 | `krabs-v3.safetensors` | 1.0 | `eugene krabs` | 训练集含“抱钱云(美元符号)”→ 负面排 `money/dollar sign/coins/money bag` 防怪道具 |
| 小蜗 | `gary-v3.safetensors` | 1.0 | `gary the snail` | 审计点：眼柄恰好两根、壳完整 |

## 3. 抠图（RMBG-2.0）+ alpha 收紧

- ComfyUI `RMBG` 节点：`model=RMBG-2.0, process_res=1024, sensitivity=1.0, mask_blur=0, mask_offset=0`，输出 Alpha。
- 后处理 `tighten_alpha(floor=115, erode=2)`：alpha 低于 115 截断、再收缩 2px —— 消除白底渗出的白色 halo（三/四代 halo 教训）。
- 抠图后按 bbox 裁到内容包围盒，贴入比例以**裁剪后高度**为分母计算。

## 4. 背景与合成

### 4.1 官方图案背景（`bg-flat.py`，程序绘制，非生成）

- 底色深绿 `(64,99,43)`、沙地 `(216,196,140)`、沙地描边 `(88,104,44)`——全部采样自样张 `target-ref.jpg`。
- 花朵云形轮廓 11 朵沿边缘与上半部排布（样张同款位置、6 色、高斯腐蚀描边不产生凹角伪影），中部角色区留空；
  白色小气泡圈散布上半部；底部沙地椭圆出血。
- **背景直接画在 1440x3200 画布上**，成品原生 1440x3200，**全程无放大环节**（C4 的分辨率保证来自这里，不走 Remacri）。

### 4.2 贴入规则（C3“别放大过度”的硬约束）

- **贴入比 = 贴入高 / 单元裁后高 ≤ 1.0x，只允许缩小贴入**（“裁后高”= alpha 收紧并 bbox 裁剪后的单元高度；`compose-round8.py` 逐角色打印该比值，审计报告核对）。
  原因：放大贴入是六代海绵宝宝“发软 + 肢体崩”的直接诱因；缩小贴入等于超采样，线条只增不减。
  （注：七代 manifest 写的 0.62x 是按未裁全高 1300px 口径；按本配方统一口径，七代胜出单元实测 ≈0.93x，同样 ≤1.0x。口径以本配方为准。）
- 官方比例（1440 宽画布下角色全高，海绵宝宝=基准）：

| 角色 | 全高 px | 相对 | 贴入比实测（八代） |
|---|---|---|---|
| 海绵宝宝 | 780–810 | 1.00 | 0.92–0.94x |
| 珊迪 | ~1000 | ≈1.25 | ≤0.8x |
| 派大星 | 970–990 | ≈1.22 | 0.81–0.82x |
| 蟹老板 | ~1050–1080 | ≈1.32 | ≤0.85x |
| 章鱼哥 | 1120–1150 | ≈1.40 | ≤1.0x |
| 小蜗 | 300–330（前景地面） | — | 大幅缩小，天然安全 |

- 站位/倾角：后排 `ground_y≈2550–2610`，前排 `2820–2900`；±3–5° 倾角制造动势；z 序 后→前 显式声明。
- **C2 互动靠三件事**：① 视线方向（各单元生成时用 looking-to 对齐对手）；② 前后叠压与间距（集群围合，不并列）；③ 泡泡带——程序绘制样张同款线环泡泡+小实心点，从一侧角色升起、掠过中间角色、落到另一侧伸出的手/头，串起视线链。**除泡泡外不加任何道具**（C4 防怪道具）。

### 4.3 派大星固定后处理三件套（顺序不可换）

1. `clean_patrick_blue`：短裤蓝色彩条/花形孤岛（Patrick_il 属性泄漏）→ 局部面板色调和扩散回填；
2. `clean_patrick_blotches`：绿短裤上被绿色围合的异色小孤岛 → 短裤中位色回填（BFS 围合判定，防误伤眼/描边）；
3. `fix_patrick_outline`：外轮廓 10px 带内强红像素（权重=红度×低绿度）向近黑 `(20,18,16)` 均匀混合——
   描边核心 (227,85,70) 全量回色、皮肤粉不动、口腔红在带外不受影响。
（顺序原因：先 blue 再 blotches，否则 blotches 级联留橄榄 AA 圈、blue 空转。参数即 PR #14/#16 固化值。）

## 5. 审计规程（单元级 + 成品级）

单元级（选种子时，逐张看）：肢体齐全（海绵宝宝双臂双手四指双腿恰好两鞋）、无第三人/人群、无道具、
无伪影（绿箭头、粉块、三只鞋、前景巨手）、未被裁切、视线方向正确。
→ 七代 manifest 记录 22 个失败种子及原因，全部能被以上条目拦下。

成品级（1440x3200 原分辨率 + 关键区域 100% 裁切复核）：

| 判据 | 检查 |
|---|---|
| C1 | 与样张并排比：底色/花朵/沙地同款、角色为官方画风、无 3D/写实感 |
| C2 | 角色数 ≤3；视线两两相关、有前后叠压与共同焦点；泡泡带存在（如用） |
| C3 | 海绵宝宝双臂双手双腿、逐指可辨、恰好两只鞋；所有角色无缺件；贴入比 ≤1.0x（脚本打印报告核对） |
| C4 | 描边无偏色（尤其派大星外缘）、无漂浮怪道具、无文字/水印、无畸变、尺寸恰 1440x3200 |

任何一条不过 → 该张判不合格，记录路径与具体失败原因（进 issue 报告），不合格图不进 `delivery/batch/`。

## 6. 复现步骤

```bash
# ① 单元出图（ComfyUI 需在 127.0.0.1:8188，模型见 §1/§2）
python3 gen-round8.py char --role sb-left  --seeds <s1>,<s2>,...   # 每角色 3–5 seed
# ② 人工按 §5 单元审计选种子后抠图
python3 gen-round8.py rmbg --role sb-left  --keep <选中文件>
# ③ 背景
python3 bg-flat.py --out <bg.png>                                   # 1440x3200
# ④ 合成（场景表在 compose-round8.py 的 SCENES 内，逐场景一条命令）
python3 compose-round8.py --bg <bg.png> --scene <scene-id> --out <out.png>
# ⑤ 按 §5 成品审计；合格图收进 delivery/batch/
```
