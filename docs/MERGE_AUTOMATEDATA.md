# Merge `origin/automateData` เข้า `setup_env`

สถานะ ณ 2026-10-01: **merge ยังไม่ commit** (`git status` ยังเป็น merge in progress, รอเจ้าของตรวจ) —
เอกสารนี้สรุปว่า merge อย่างไร เจอปัญหาอะไร แก้อย่างไร และผลทดสอบอ้างอิงอยู่ที่ไหน
commit ก่อน merge คือ `99e39f6` (= `origin/setup_env`); commit ที่ merge เข้ามาคือ `b9c7ca6` (`origin/automateData`)

## 1. สรุปสั้น

| หัวข้อ | ผล |
| --- | --- |
| unit test | ผ่าน 43/43 (40 เดิม + 3 ของ `test_place_insert.py` ที่ปรับให้เข้ากับฉากใหม่) — รันล่าสุดหลังแก้ `PlugGeometry` |
| expert แบบถือปลั๊ก (`validate_openarm.py`, v1) | **เหมือนก่อน merge ทุกตัวเลข** ทั้ง 30 episode — `results/openarm_cable_v2_merge` เทียบ `results/openarm_cable_v2` |
| งานเต็มหยิบจากโต๊ะ (`probe_table_insert.py`, v1) | ผ่าน, depth 0.015988779633931693 m **ตรงกับก่อน merge ทุกบิต** — `results/full_task_cable_v1_merge` |
| automation (`collect_random.py`) บน v1 | รอบแรก **0/4** → **5/15** → หลังแก้ทั้งหมด **10/16** (ข้อ 1.1, 5) |
| automation บน v2 (Type O, สาย 1.5 m) | **3/3** หลังให้ expert อ่านขนาดปลั๊กจาก spec (ข้อ 4.10, 6) — `results/automate_merge_ws2_type_o` |

### 1.1 ปรับอะไร ผลก่อน/หลัง (automation บน workspace_v1, detector privileged)

แต่ละขั้นรวมทุกอย่างของขั้นก่อนหน้า

| ขั้น | ปรับอะไร | ผล | หลักฐาน |
| --- | --- | --- | --- |
| 0 | branch ของเขา บนฉากเก่า (ไม่มีโต๊ะจริง/สาย/ลีฟ) | ~95% ตาม log ของเขา | `docs/AUTOMATED_COLLECTION_LOG.md` |
| 1 | merge: tabletop mode บน env ใหม่, วางปลั๊กผ่าน `place_plug`, เสาย้ายตามเต้ารับ, sampler อ่านความสูงเต้ารับจาก workspace, planner ไม่นับลีฟ/สาย | **0/4** | `results/automate_merge_ws1_first4` |
| 2 | ตอนเสียบ: เมื่อลีฟแตะ ดันแบบจำกัดแรง 15 N, ไม่ใช้เพดานผนัง 5 N, ไม่ถอย | 4 ฉากเดิม: **1/4** (train_00000 สำเร็จ 16.0 mm) | `…first4/summary.json` |
| 3 | sampler: ปลั๊กต้องเริ่มหน้าเต้ารับ (สายไม่พาดเสา), เต้ารับห่างมือพัก ≥ 10 cm, ระยะสาย, ช่วงสุ่มใหม่ | preflight ฉากใหม่ **5/15** | `results/automate_merge_ws1_preflight_before_insertfix` |
| 4 | ตัวเช็คตอนหยิบใช้ `plan_insertion` ตัวเดียวกับตอนเสียบ (seed เหมือนตอนรัน) | preflight ฉากใหม่ **10/16** | `results/automate_merge_ws1_preflight_after_insertfix` |
| — | ลองเพิ่มเกนแก้ด้านข้าง ×2 | ไม่ช่วย ฉากติดยังติด แรงผนังฉากที่ผ่านขึ้น 8.7 → 13.4 N → **ถอยกลับ** | `…first4/rerun_lateral_reverted_attempt.txt` |
| — | ลองคุมเป้าหมายไม่ให้นำปลั๊กเกิน 1 mm เมื่อลีฟแตะ (lead1) | 4 ฉากที่ติด: **0/4** ติดที่เดิม → **ถอยกลับ** | `results/automate_merge_ws1_stall_variants` |
| 5 | ค่าปลั๊กที่ hardcode ใน expert อ่านจาก connector spec (`PlugGeometry`) — ค่าของ v1 เท่าเดิมทุกตัว | v1: 2 ฉากที่เคยผ่าน **2/2** (ไม่เปลี่ยน); v2 Type O: **3/3** | `results/automate_merge_ws1_geometry_regression`, `results/automate_merge_ws2_type_o` |
| — | ลองยกเพดานแรงดันเมื่อลีฟแตะ 15 → 20 N (cap20) | 4 ฉากที่ติด: **0/4** (2 ติด, 2 แรงลีฟเกิน 80 N → abort) → **ไม่นำมาใช้** | `results/automate_merge_ws1_stall_variants` |

ชนิดของความล้มเหลว

| อาการ | ขั้น 1 (4 ฉาก) | ขั้น 3 (15 ฉาก) | ขั้น 4 (16 ฉาก) |
| --- | --- | --- | --- |
| **success** | 0 | 5 | **10** |
| เสียบติดที่ลีฟ / align ไม่ได้หลังถอย | 2 | 1 | 4 |
| `plan_insert_line_infeasible` | – | 8 | **0** |
| `plan_seated_unreachable` | – | 1 | 0 |
| `ik_infeasible` (ล้มก่อนขยับ ~15 s ไม่เสียเวลา) | – | – | 2 |
| ปลั๊กหลุดมือเพราะสายพาดเสา | 1 | 0 | 0 |
| `transit_blocked` (เต้ารับชิดมือพัก) | 1 | 0 | 0 |

ที่ยังค้าง — เสียบติด 4/16 (ข้อ 4.9) มี 2 กลไก ยังไม่มีวิธีแก้ที่พิสูจน์แล้ว:
1. ที่ ~10 mm (3 ฉาก): ปลั๊กตรงแนว (< 30 µm) จนลีฟแตะ แล้ว**ตกลง ~0.36 mm ในแกน z ภายใน 0.6 s**
   ขณะแรงดันขึ้น 12 → 14 N ขาเบียดผนังล่าง (3–4 N) แรงที่ต้องใช้กลายเป็น ~17.4 N เกินเพดาน
2. ที่ 4 mm (1 ฉาก): ปลั๊กตรงแนว (< 10 µm) แต่ลีฟตัวเดียวแตะแล้วผลักปลั๊กเยื้อง 0.25 mm ทันทีแล้วกั้นไว้
   ทั้งที่ออกแบบให้ลีฟเริ่มแตะที่ ~9.9 mm — ยังไม่รู้สาเหตุ (ฉากนี้เต้ารับหมุน 19°)

ส่วนที่ต้องไม่เสีย (ผ่านทุกข้อ): unit test 43/43; `validate_openarm.py` และ `probe_table_insert.py`
หลัง merge ได้ผล**เท่ากับก่อน merge ทุกตัวเลข**

## 2. branch `automateData` มีอะไร

5 commit ของ Pokpong-S (2026-09-20 ถึง 09-25) แตกออกจาก `41413e3` ซึ่งเก่ากว่างานฉาก workspace ทั้งหมด
(โต๊ะจริง, สายไฟ, ลีฟสปริง, catalog ปลั๊ก) สิ่งที่เพิ่ม:

- `controllers/pick_insert_expert.py`, `controllers/place_insert.py` — expert หยิบปลั๊กจากโต๊ะแบบบน-ลง
  แล้วเสียบ (top-down grasp) พร้อม planner IK
- `data_pipeline/scene_bank.py` — สุ่มตำแหน่งปลั๊ก/เต้ารับบนโต๊ะ + ตรวจความเป็นไปได้ด้วย IK
- `scripts/collect_random.py` — collector อัตโนมัติ (preflight → record), `rollout_viewer.py`,
  `replay_episode.py`, collector แบบ manual สองตัว
- `envs/openarm_insert.py` +351 บรรทัด — โหมด `tabletop` (ปลั๊กวางบนโต๊ะ, ย้ายเต้ารับได้), grasp latch,
  นับ drop, แยกแรงสัมผัสโต๊ะ, `apply_visual`
- `docs/AUTOMATED_COLLECTION_LOG.md` — log ของเขา (รายงาน ~95% สำเร็จ **ในฉากเก่า**)

## 3. วิธี merge

ใช้ `git merge --no-ff --no-commit origin/automateData` (ไม่ rebase เพื่อไม่ต้อง force-push `setup_env`)

| ไฟล์ | conflict | วิธี resolve |
| --- | --- | --- |
| `envs/openarm_insert.py` | ใช่ | เริ่มจากของเรา แล้ว port tabletop mode ของเขาเข้ามาทีละส่วน (ข้อ 4.1–4.4) |
| `envs/scene.py` | ใช่ (เขา comment เสา fixture ทิ้งในตัวสร้างฉากเก่า) | ใช้ของเรา — ฉากมาจาก workspace JSON และเสาย้ายตามเต้ารับแทน |
| `.gitignore` | ใช่ (เขา ignore `CLAUDE.md`) | เก็บ `.claude/` ไม่ ignore `CLAUDE.md` เพราะ branch นี้ track ไฟล์นั้น |
| `configs/openarm_v1.json` | auto-merge | คง `episode_limit_s: 8` ของโหมดถือปลั๊ก เพิ่ม `tabletop_episode_limit_s: 240` และ `grip_open_travel_m: 0.044` |
| `data_pipeline/episodes.py` | auto-merge | รับของเขา (ให้ episode `human_manual` ผ่าน validate) |
| `results/two_blade_v1_straight/*` | auto-merge (เขาแก้หลักฐานเดิม) | **คืนไฟล์เดิม 4 ไฟล์** (ห้ามแก้ผลที่ docs อ้างถึง) เก็บเฉพาะ `combined/` และ `results/full_task_new/` ที่เพิ่มใหม่ |

`envs/openarm_insert_temp.py` ของเขาเข้ามาด้วยแต่ไม่มีไฟล์ไหน import — ปล่อยไว้ ไม่ได้ลบ

## 4. ปัญหาที่เจอและวิธีแก้

### 4.1 โค้ดของเขาสมมติฉากเก่า (ก่อน workspace)

| ปัญหา | แก้ |
| --- | --- |
| วางปลั๊กด้วยการเขียน qpos ตรง ไม่วางสายไฟตาม (CLAUDE.md กำหนดให้ใช้ `place_plug` เท่านั้น) | tabletop reset เรียก `env.place_plug(...)` ซึ่งวางสายจากปลั๊กไปกล่องให้ถูก |
| ย้ายเต้ารับไปวางบนโต๊ะ แต่ฉากใหม่เต้ารับติดอยู่บนเสา fixture สูงจากโต๊ะ 158 mm | ย้ายเสาไปพร้อมเต้ารับด้วย rigid transform เดียวกัน (`carry_fixture()` ใน `envs/openarm_insert.py`) ความสูงเต้ารับคงตาม workspace |
| model ถูกแก้ค้าง (ตำแหน่งเต้ารับ/เสา, กล้อง, แสง, สีโต๊ะ) ข้าม reset | `_restore_scene()` คืนค่าทุกอย่างก่อนทุก reset → โหมดถือปลั๊กและ replay เห็นฉากเดิมเสมอ |
| ใช้ config key ที่ branch นี้เปลี่ยนเป็นค่าคำนวณแล้ว (`initial_finger_travel_m`, `grip_target_travel_m`) | ใช้ `finger_contact_travel` / `grip_travel` ที่คำนวณจากความกว้างตัวเรือนปลั๊ก |
| `apply_visual` หา geom ชื่อ `table` (ฉากใหม่ชื่อ `work_table`) | แก้ชื่อ |

### 4.2 เขาเปลี่ยนพฤติกรรมโหมดถือปลั๊กด้วย

`step()` ของเขาเปลี่ยนเงื่อนไข abort ของทุกโหมด (slip/rotation นับเฉพาะหลัง latch grasp, เพิ่ม
`table_collision`, แยกแรงสัมผัสโต๊ะออกจาก unwanted contact) และ reset วางแขนซ้ายที่ศูนย์ทุกโหมด
→ ใช้ logic ของเขา **เฉพาะโหมด tabletop** โหมดถือปลั๊กคงของเดิมทุกบรรทัด

หลักฐาน: `validate_openarm.py` หลัง merge (`results/openarm_cable_v2_merge`) ได้ checks, outcome,
depth, peak force และ duration **เท่ากันทุก episode** กับ `results/openarm_cable_v2` (9/10 เหมือนเดิม
ข้อที่ไม่ผ่านคือ `half_timestep_force` ข้อเดิม) และ `results/full_task_cable_v1_merge` depth ตรงกันทุกบิต

### 4.3 sampler และ feasibility check ไม่รู้จักฉากใหม่

| ปัญหา | แก้ |
| --- | --- |
| สุ่มเต้ารับที่ความสูงวางบนโต๊ะ (entry สูงกว่าโต๊ะ 15.2 mm) | `workspace_for_env()` อ่านความสูงเต้ารับจริงจาก env (158 mm บน v1, 152 mm บน v2) |
| `PickFeasibility` ย้ายเต้ารับใน model ส่วนตัวแต่ไม่ย้ายเสา | เรียก `carry_fixture()` เหมือน env |
| `InsertFeasibility` สร้าง env v1 เสมอ | รับ `workspace=` และ cache แยกตาม workspace |
| ไม่มีข้อจำกัดความยาวสาย | เพิ่ม `cable_reach`: จุดต่อสายตอนหยิบและตอนเสียบต้องห่าง anchor ≤ 0.95×ความยาวสาย |
| cache ของ scene ไม่แยกตามฉาก | field ใหม่ใน `Workspace` (path, sha256, ความสูง, anchor) อยู่ใน cache key |

### 4.4 collision check ของ planner ไม่รู้จักลีฟและสาย

ทุกฉากล้มด้วย `plan_seated_collision` แม้ตำแหน่งเต้ารับเดิม — contact ที่ชนคือ
`blade_left|socket_leaf_left`, `blade_right|socket_leaf_right` (ลีฟยื่นเข้าช่อง 0.25 mm ออกแบบให้แตะปลั๊ก)
→ นับ body ของลีฟเป็นเต้ารับ และไม่นับ contact กับสายใน planner ทั้งสองตัว (ตาม contract เดิมที่
`plan_carry` ไม่สนสาย) — `controllers/place_insert.py` `ArmKinematics`, `controllers/pick_insert_expert.py`
`GraspPlanner.contacts`

### 4.5 test ของเขา

`tests/test_place_insert.py` วางปลั๊กที่ (0.50, 0.25) ห่าง anchor สาย 0.67 m (สายยาว 0.35 m →
`cable too short`) และคาดว่าเต้ารับวางบนโต๊ะ → ปรับให้ปลั๊กอยู่ในระยะสาย, ตรวจว่าเต้ารับยังติดเสา
(เสาเลื่อนด้วย transform เดียวกัน) และ reset แบบถือปลั๊กคืนตำแหน่งเดิม; ฉากที่เสาไปชนมือตอนพัก
reset จะปฏิเสธ (sampler ก็ตัดทิ้งเหมือนกัน)

### 4.6 automation บน v1 รอบแรก 0/4 — วิเคราะห์ทีละฉาก

ผล: `results/automate_merge_ws1_first4/` (`summary.json`, JSON ของ collector, ภาพทุก phase)

| ฉาก | อาการ | สาเหตุ (วัดได้) | แก้ |
| --- | --- | --- | --- |
| train_00000 | เสียบติดที่ 10.45 mm แล้วถอยแล้ว align ไม่ได้ | controller เสียบของเขาทำมาสำหรับเต้ารับแข็ง: stall 1 s = ล้ม, แรงผนัง > 5 N = ถอย, ถอยได้ — แต่ลีฟต้องดันผ่าน ~10 N และยึดปลั๊กไว้ | `place_insert.py`: เมื่อลีฟแตะแล้ว ดันแบบจำกัดแรง `push_force_cap_n` (15 N), ไม่ใช้เพดาน 5 N, ไม่ถอย (แบบเดียวกับที่แก้ held-plug expert) → **สำเร็จ** เสียบ 16.00 mm แรงผนังสูงสุด 8.73 N |
| train_00001 | `transit_blocked` ก่อนขยับ | ท่าพักของหุ่นวางมือตรงตำแหน่งเสียบของ workspace (0.401, −0.155, 0.477) sampler สุ่มเต้ารับห่างมือ 8 mm นิ้วจม lead-in 2–5 mm ตั้งแต่ขั้นแรกของ transit | กฎ sampler `socket_at_hand_home`: เต้ารับต้องห่างมือพัก ≥ 0.10 m |
| train_00002 | ปลั๊กหมุนในมือ 5.7° ระหว่างขน | ปลั๊กเริ่ม**หลัง**หน้าเต้ารับ พอหิ้วอ้อมไปหน้าเต้ารับ สายพาดข้ามยอดเสา แรงดึงสาย 0.1 → 23.7 N (contact สาย–เสา ท่อน 00–08) — ไม่ใช่เรื่องสายสั้น (ระยะตรงสูงสุด 0.284 จาก 0.3325 m) | กฎ sampler `cable_wraps_pedestal`: ปลั๊กต้องเริ่มฝั่งหน้าเต้ารับ ≥ 30 mm |
| train_00003 | เสียบติดที่ 11.3 mm | ขาปลั๊กเบียดผนังรูตอนลีฟเริ่มกด (แรงดันตามแกน 17 N, เยื้อง 0.54 mm) สายหย่อน (0.12 N) ระหว่างเสียบ | **ยังแก้ไม่ได้** ลองเพิ่มเกนด้านข้าง ×2 แบบ probe แล้ว ฉากนี้ยังติด (9.4 mm) และแรงผนังฉาก 0 ขึ้นจาก 8.7 → 13.4 N จึง**ถอยกลับ** ฉากนี้ปลั๊กอยู่หลังเต้ารับ กฎ `cable_wraps_pedestal` ตัดทิ้งแล้ว |

### 4.7 ช่วงสุ่มเดิมแคบเกินไปหลังมีกฎใหม่

preflight รอบที่สองพังตอนสร้างชุด evaluation: seed 9000015 ไม่เจอฉากที่ใช้ได้ใน 200 ครั้ง ช่วงเดิมผ่าน
เช็คเบื้องต้นแค่ 3.8% (เต้ารับสูง + กฎใหม่) → เฉพาะฉาก workspace ใช้ช่วงใหม่ เต้ารับ x 0.38–0.46,
y −0.36…−0.20; ปลั๊ก x 0.28–0.38 → ผ่านเช็คเบื้องต้น 51.5% และ IK 13/25 (ช่วงของฉากโต๊ะเปล่าเดิมไม่เปลี่ยน)

### 4.8 ตัวเช็คของ pick planner กับ insertion planner ตัดสินไม่ตรงกัน

preflight บนฉากใหม่ (หลังแก้ 4.1–4.7) ได้ **5/15** — `results/automate_merge_ws1_preflight_before_insertfix/`
(`preflight_log.txt`, JSON ของฉากที่ล้ม, ภาพทุก phase ของ 5 ฉากที่สำเร็จ)

| อาการ | จำนวน |
| --- | --- |
| success | 5 |
| `plan_insert_line_infeasible` | 8 |
| `insert_stalled` (4.02 mm, ลีฟตัวเดียวรับ 17.7 N, ปลั๊กเยื้อง 0.36 mm) | 1 |
| `plan_seated_unreachable` | 1 |

วิเคราะห์ (train_00005, train_00006): ท่าจับที่หยิบจริงตรงกับท่าจับมาตรฐาน (ต่าง 0.1 mm, 0°) และแม้ใช้
ท่าจับมาตรฐานเริ่มจากท่าอ้างอิงก็ยังวางแผนเสียบไม่ได้ → ปัญหาไม่ใช่การจับ แต่ pick planner เลือก tilt −15°
โดยใช้ตัวเช็ค "เสียบได้" ของตัวเอง (เช็คแค่ 2 ท่า: standoff 55 mm และท่าเสียบ) ซึ่งหลวมกว่า
`place_insert.plan_insertion` ที่ใช้จริง (ต้องเดินเส้นตรงเข้ารูได้) ส่วน sampler ผ่านเพราะมีท่าจับ**อื่น**ที่เสียบได้

แก้:
- `GraspPlanner.insert_feasible` ต้องผ่าน `plan_insertion` ตัวเดียวกับตอนเสียบจริงด้วย — pick, sampler
  และการเสียบใช้เกณฑ์เดียวกัน
- `PlaceInsert.start()` ลองวางแผนจากท่าปัจจุบันก่อน แล้วจาก `REFERENCE_POSTURE`
- ตัวเช็คใช้ seed ชุดเดียวกับตอนรันจริงเรียงเหมือนกัน: ท่าหลังยกปลั๊กของท่าจับนั้น แล้วจึง `REFERENCE_POSTURE`
  — ครั้งแรกใช้แค่ `REFERENCE_POSTURE` แล้วเข้มเกินจริง: ผ่าน IK แค่ 0/20 และ 1/20 (ช่วง A/B) จน
  preflight พังตอนสร้างชุด evaluation (seed 9000000 ไม่มีฉากใน 200 ครั้ง) เพราะ IK ท่าเสียบที่เต้ารับสูง
  ขึ้นกับ seed มาก; พอใช้ seed ตามจริงได้ 3/20 ทั้งสองช่วง (ก่อนแก้ข้อนี้ sampler ผ่าน 13/25 แต่ส่วนใหญ่ไปล้มตอนรัน)

ยืนยัน: รัน train_00005/00006 ซ้ำ → planner ไม่เลือกท่าจับที่เสียบไม่ได้แล้ว ล้มทันทีที่ `ik_infeasible`
(13–14 s ก่อนขยับ) แทนที่จะหยิบแล้วค่อยล้ม และ sampler จะตัดฉากแบบนี้ทิ้งตั้งแต่ตอนสุ่ม

### 4.9 เสียบติดที่ยังเหลือ (วิเคราะห์ ยังไม่แก้)

trace ละเอียดช่วง align/insert: `results/automate_merge_ws1_stall_variants/trace_train_00007.txt`,
`trace_train_00000.txt`

- การจัดแนวก่อนเสียบ**ไม่ใช่**ปัญหา: ตอนจบ align และตลอดทางถึง 9.9 mm ปลั๊กเยื้อง < 30 µm
  (ที่เคยเสนอว่าบีบเกณฑ์จัดแนว 0.3 → 0.1 mm จึงตัดทิ้ง)
- train_00007: ลีฟแตะที่ 9.9 mm แรงดันขึ้น 3 → 14 N, ปลั๊กตกจาก z −27 → −363 µm ภายใน 0.6 s
  (ระยะว่างขา–ผนัง ~0.4 mm) ปลั๊กลื่นในมือเพิ่ม 0.02 → 0.145 mm — น่าจะเป็นข้อมือที่ยืดหยุ่น
  (kp 40 Nm/rad) รวมกับการจับด้านบนที่แรงตามแกนสร้างแรงบิดรอบแกนบีบ (ยังเป็นสมมติฐาน)
- ลองแล้วไม่ได้ผล: เกนด้านข้าง ×2, คุมเป้าหมายนำปลั๊ก ≤ 1 mm, เพดานดัน 20 N (ตารางข้อ 1.1)

### 4.10 expert ของ automateData ผูกกับปลั๊ก 2 ขา

ค่าปลั๊กราว 30 จุดใน `pick_insert_expert.py`/`place_insert.py` เป็นค่าคงที่ของปลั๊ก 2 ขา
และโค้ดถือว่าหมุนปลั๊ก 180° รอบแกนเสียบแล้วยังเสียบได้ (ขา 2 ขาเท่ากัน) — ใช้กับ Type O ไม่ได้

แก้: `PlugGeometry` + `plug_geometry(model, workspace)` ใน `controllers/place_insert.py` คำนวณจาก
connector spec แล้วส่งเข้า `GraspPlanner`, `PickFeasibility`, `PickInsertExpert`, `InsertFeasibility`,
`PlaceInsert`; `scene_bank.workspace_for_env` อ่านความสูงปลั๊กบนโต๊ะจาก spec

| ค่า | ค่าเดิม (hardcode) | จาก spec: v1 | จาก spec: v2 Type O | ที่มา |
| --- | --- | --- | --- | --- |
| จุดกำเนิดปลั๊กเหนือโต๊ะ | 8 mm | 8 mm | 4 mm | `table_rest_z_offset_m` |
| จุดจับ (กลางตัวเรือน) | (−16, 0, 0) mm | เท่าเดิม | (−16, 0, 6) mm | `plug.grasp_offset_m` |
| นิ้วเปิดก่อนจับ | 30 mm | 30 mm | 35 mm | ระยะแตะ + 16.2 mm |
| นิ้วสั่งปิด | 6 mm | 6 mm | 11 mm | ระยะแตะ − `pickup_squeeze_m` |
| นิ้วตอนบีบแตะตัวเรือน | 13.7 mm | 13.7 mm | 18.7 mm | ระยะแตะ − 0.1 mm |
| standoff ของ pick planner | 55 mm | 55 mm | 60.4 mm | +(ขายาวขึ้น) |
| ระยะรอหน้ารู | 25 mm | 25 mm | 30.4 mm | `probe_preplug_x_m` |
| หมุน 180° แล้วเสียบได้ | ได้ | ได้ | **ไม่ได้** | `symmetry_rolls_deg` |

ยืนยัน: ค่าที่คำนวณได้ของ v1 ตรงกับค่าเดิมทุกตัว (ตรวจทีละค่า), unit test 43/43, 2 ฉาก v1 ที่เคยผ่าน
ยังผ่าน (16.00 / 15.99 mm)

## 5. ผล automation บน v1 หลังแก้ทั้งหมด

`results/automate_merge_ws1_preflight_after_insertfix/` (log, ฉากที่ผ่าน, JSON และภาพทุก phase ของฉากที่ล้ม)

- **10/16 สำเร็จ** (62%) detector privileged, 10 ฉากสำเร็จครบเป้า
- ล้ม 6: เสียบติด 4 (ข้อ 4.9), `ik_infeasible` 2 (planner หาท่าหยิบไม่ได้ตั้งแต่ก่อนขยับ ทั้งที่ sampler ผ่าน —
  ตัวเช็คสองฝั่งยังไม่ตรงกันในบางฉาก แต่ล้มเร็ว ~15 s)
- ไม่เจอ `plan_insert_line_infeasible` อีก (ขั้น 3 เจอ 8/15)
- เวลา ~10 นาที/ฉาก (wall-clock) บนเครื่องนี้; สร้าง scene cache ใหม่ ~1 ชม. ทุกครั้งที่แก้ controller/sampler
- **ยังไม่ได้**: detector noisy, `--record` + replay แบบเข้มงวด

## 6. automation บน v2 (Type O, สาย 1.5 m)

`results/automate_merge_ws2_type_o/` (ฉากที่สุ่ม, summary ของแต่ละ episode, ภาพทุก phase, สคริปต์ที่ใช้รัน)

หลังข้อ 4.10: **3/3 สำเร็จ** (detector privileged, ฉาก seed 1000000–1000002 สุ่มด้วย sampler ปัจจุบัน)

| ฉาก | ผล | ลึก | แรงผนังสูงสุด | เวลาจริง |
| --- | --- | --- | --- | --- |
| v2_00000 | success | 18.83 mm | 12.1 N | 46 นาที |
| v2_00001 | success | 18.86 mm | 18.6 N | 36 นาที |
| v2_00002 | success | 18.84 mm | 9.1 N | 61 นาที |

(เกณฑ์สำเร็จ Type O คือ 18 mm; เพดาน abort ผนังของ v2 คือ 30 N)

ข้อจำกัด: แค่ 3 ฉาก ยังไม่ใช่อัตราสำเร็จที่เชื่อถือได้; ช้ามาก (~45 นาที/ฉาก บนเครื่องนี้ เพราะสาย 60 ท่อน);
ยังไม่ได้รันผ่าน `collect_random.py` เต็มรูปแบบ (evaluation bank ของ v2 ยังไม่ได้สร้าง), ยังไม่ได้ลอง noisy/record

## 7. ไฟล์ที่เปลี่ยนเทียบกับไฟล์ที่ได้จาก branch เขา

- `envs/openarm_insert.py` — merge มือ: tabletop mode, `carry_fixture`, `_restore_scene`, grasp latch เฉพาะ tabletop
- `controllers/place_insert.py` — ลีฟ/สายใน collision check, เสียบแบบรู้จักลีฟ, `InsertFeasibility(workspace=)`
- `controllers/pick_insert_expert.py` — ไม่นับ contact กับสาย, `PickFeasibility(workspace=)` + ย้ายเสา,
  `insert_feasible` ต้องผ่าน `plan_insertion` ด้วย
- `controllers/place_insert.py` `PlaceInsert.start()` — ลอง seed `REFERENCE_POSTURE` ถ้าท่าปัจจุบันวางแผนไม่ได้
- `controllers/place_insert.py` `PlugGeometry`/`plug_geometry()` และทั้งสองไฟล์ใช้ค่าปลั๊กจาก spec (ข้อ 4.10)
- `data_pipeline/scene_bank.py` — ค่าจาก workspace, `cable_reach`, `cable_wraps_pedestal`, `socket_at_hand_home`, ช่วงสุ่มใหม่
- `scripts/collect_random.py`, `scripts/replay_episode.py` — `--workspace`
- `tests/test_place_insert.py` — ปรับตามฉากใหม่
- `configs/openarm_v1.json`, `.gitignore` — ตามข้อ 3

⚠️ `envs/openarm_insert.py` และ `configs/openarm_v1.json` อยู่ใน `env.manifest()` → dataset เดิม replay
แบบเข้มงวดไม่ได้ (ต้องเก็บใหม่; เป็นอยู่แล้วตั้งแต่ workspace_v1)

## 8. ยังไม่ได้ทำ / ข้อจำกัด

- ฉากแบบ train_00003 (เสียบติดตอนลีฟแตะด้วย grasp แบบบน-ลง) ยังไม่มีทางแก้
- เต้ารับบนเสาสูงทำให้พื้นที่สุ่มแคบลงมาก เทียบกับฉากเก่าของเขา
- ยังไม่ได้ record + replay ข้อมูลจาก automation บนฉากใหม่
- ตัวเลขทั้งหมดเป็น simulation, detector แบบ privileged เท่านั้น (ยังไม่ได้ลอง noisy)
- v2: ทดสอบแค่ 3 ฉาก (ช้า ~45 นาที/ฉาก) ยังไม่ได้รัน `collect_random.py --workspace configs/workspace_v2.json` เต็มรูปแบบ
- ฉากที่ลีฟกั้นที่ 4 mm (train_00000) ยังไม่รู้สาเหตุ

## 9. รันซ้ำ

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
.venv\Scripts\python.exe scripts/validate_openarm.py --output results/<ใหม่>
.venv\Scripts\python.exe scripts/probe_table_insert.py --output results/<ใหม่>
.venv\Scripts\python.exe scripts/collect_random.py --episodes 10 --early-abort 6 --headless --video all --detector privileged --output data/<ใหม่>
.venv\Scripts\python.exe scripts/collect_random.py --workspace configs/workspace_v2.json ...   # v2
```

ครั้งแรกหลังแก้ controller/sampler จะสร้าง scene cache ใหม่ (~1 ชม. สำหรับ evaluation bank 100 ฉาก)
