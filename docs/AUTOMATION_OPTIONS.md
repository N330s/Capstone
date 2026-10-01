# เลือกแนวทาง automation บนฉาก setup_env

เอกสารนี้เทียบ 2 แนวทางสำหรับเก็บ data อัตโนมัติบนฉากของ `setup_env` (v1 ปลั๊ก 2 ขาแบน และ v2 Type O)
ตามโจทย์: **สุ่มทั้งปลั๊กและเต้ารับ, ปลั๊กเริ่มจากบนโต๊ะด้านล่าง สุ่มเฉพาะตำแหน่งหัวปลั๊ก** (สายและกล่อง appliance อยู่ตาม workspace)

ตัวเลขทั้งหมดมาจาก `results/` หรือ log ของรอบที่รันจริง; ของที่ยังไม่ได้ทดสอบเขียนไว้ว่ายังไม่ได้ทดสอบ

## สองแนวทาง

| | **A. controller ของ setup_env** | **B. expert ของ automateData** |
| --- | --- | --- |
| ไฟล์หลัก | `controllers/table_pickup.py` (`PickupProbe`), `controllers/carry_path.py` (`plan_carry`), การเสียบใน `scripts/probe_table_insert.py` | `controllers/pick_insert_expert.py`, `controllers/place_insert.py` |
| ท่าเริ่มหุ่น | แขนทั้งสองห้อยลง (ท่า zero, มือคว่ำลง) | แขนขวายกค้างที่ท่า home (มืออยู่ตรงตำแหน่งเสียบของ workspace) |
| วิธีจับปลั๊ก | จับด้านข้าง บีบหน้าตัวเรือน (24 mm บน v1 / 34 mm บน v2) | จับจากด้านบน (top-down) |
| ขนปลั๊ก | RRT (`plan_carry`) พร้อมตรวจชนของปลั๊กที่ถืออยู่ | IK + transit ของ planner เขา |
| การเสียบ | Jacobian servo ปิดลูป: แก้ด้านข้าง ×2 + anti-wind-up นับจากสัมผัส, ดันแบบจำกัดแรง 15 N | servo ของเขา + ส่วนที่แก้ตอน merge (ดันแบบจำกัดแรงเมื่อลีฟแตะ) |
| ขนาดปลั๊ก | อ่านจาก connector spec (`env.derived`) → ใช้กับ Type O ได้เลย | เดิม hardcode ปลั๊ก 2 ขา — **แก้แล้ว** อ่านจาก spec ผ่าน `PlugGeometry` (`docs/MERGE_AUTOMATEDATA.md` ข้อ 4.10) |
| บันทึก action ผ่าน `env.step` | **ไม่** — probe เขียน `env.target` แล้วเดิน physics เอง (มี `command_trace` สำหรับ replay แต่ไม่ใช่รูปแบบ episode สำหรับเทรน) | **ใช่** — episode, checksum, replay ครบ |
| collector อัตโนมัติ | ไม่มี (เป็น probe รันทีละครั้ง) | มีครบ: `collect_random.py` (preflight → record), `scene_bank.py`, detector แบบ noisy, วิดีโอ, `replay_episode.py` |

## หลักฐานที่มี

| | A. setup_env | B. automateData (หลัง merge) |
| --- | --- | --- |
| v1 หยิบปลั๊ก | 8/8 (`results/table_pickup_cable_v1`) แต่เลื่อนปลั๊กแค่ ±2 mm | ส่วนหนึ่งของงานเต็มด้านล่าง |
| v1 งานเต็ม | ผ่าน ที่จุดเดียว (`results/full_task_cable_v1_merge`, seated 15.99 mm) | สุ่มฉาก: **10/16** (`results/automate_merge_ws1_preflight_after_insertfix`) |
| v2 หยิบปลั๊ก | 8/8 กับสาย 1.0 m (`results/table_pickup_v2_longcable`) | ส่วนหนึ่งของงานเต็มด้านล่าง |
| v2 งานเต็ม | ผ่าน ที่จุดเดียว สาย 1.5 m ทั้ง timestep ปกติและครึ่ง (`results/full_task_v2_type_o_cable150_fix`, `…_halfdt`) | สุ่มฉาก: **3/3** หลังแก้ `PlugGeometry` (`results/automate_merge_ws2_type_o`) — แค่ 3 ฉาก, ~45 นาที/ฉาก |
| ตำแหน่งสุ่ม | **ยังไม่ได้ทดสอบ** เต้ารับที่ย้าย/หมุน และปลั๊กที่ห่างจุด spawn เกิน 2 mm | ทดสอบแล้ว (ข้อจำกัดดูด้านล่าง) |
| เวลาต่อ episode (sim) | ~43 s | ~30–45 s |

ความล้มเหลวของ B ใน preflight รอบล่าสุด (16 ฉาก): เสียบติดตอนลีฟรับแรง **4**,
`ik_infeasible` ก่อนขยับ **2** — ลองแก้เสียบติด 3 วิธี (เกนด้านข้าง ×2, คุมเป้าหมายนำ ≤ 1 mm, เพดาน 20 N) ไม่ได้ผลทั้งหมด — รายละเอียดใน `docs/MERGE_AUTOMATEDATA.md`

## ข้อดี ข้อเสีย

**A. controller ของ setup_env**
- ✅ ผ่านงานเต็มทั้ง v1 และ v2 แล้ว รวม v2 สาย 1.5 m ซึ่งยากสุด
- ✅ การเสียบมีการแก้ด้านข้าง ×2 ที่ทำไว้สำหรับอาการ "เยื้องแล้วติดที่ลีฟ" ซึ่งเป็นปัญหาหลักของ B ตอนนี้
- ✅ ขนาดปลั๊กมาจาก spec ใช้ได้ทั้งสองรุ่นโดยไม่ต้องแก้
- ✅ ท่าเริ่มแขนห้อยลง ไม่มีปัญหามือชนเต้ารับตอนเริ่ม (ทำให้ B ต้องกันเต้ารับให้ห่างมือ 10 cm)
- ❌ ยังไม่ได้ทดสอบกับตำแหน่งสุ่ม — IK ของการจับด้านข้างและทางขน RRT อาจล้มเมื่อเต้ารับย้าย/หมุน
- ❌ ไม่ผ่าน `env.step` ต้องย้ายลูปมาใช้ `env.step` ก่อนถึงจะบันทึก episode สำหรับเทรนได้
- ❌ ไม่มี collector ต้องต่อเข้ากับ collector/sampler

**B. expert ของ automateData**
- ✅ มี collector ครบ (สุ่มฉาก, preflight, record, replay, วิดีโอ, detector แบบ noisy) และบันทึกผ่าน `env.step` อยู่แล้ว
- ✅ ทำงานกับฉากสุ่มบน v1 ได้แล้วราวครึ่งหนึ่ง
- ❌ ~1/3 ของฉากเสียบติดเพราะเยื้อง ยังไม่มีวิธีแก้ที่พิสูจน์แล้ว
- ✅ v2 ใช้ได้แล้วหลังอ่านขนาดปลั๊กจาก spec (3/3 ฉาก) — เดิมต้องแก้หลายจุด (ขนาดปลั๊ก, ขาดิน, ระยะเข้าหา)
- ❌ ท่าเริ่มยกแขนค้าง ต่างจากงานเต็มของ setup_env

## ทางเลือกที่แนะนำ: C. ผสม

ใช้ **โครงของ B** (`collect_random.py`, `scene_bank.py`, episode/replay ผ่าน `env.step`) แต่เปลี่ยน
**ตัวควบคุมเป็นของ A** (แขนห้อยลง → จับด้านข้าง → `plan_carry` → การเสียบของ probe)

ต้องทำ:
1. เขียน expert ตัวใหม่ที่ห่อ `PickupProbe` + `plan_carry` + การเสียบของ probe ให้ออก action ทีละ step
   ผ่าน `env.step` (แทน `physics_step` เอง) — ต้องตรวจว่าผลยังเท่า probe เดิม
2. tabletop reset: แขนห้อยลงที่ท่า zero, ปลั๊กวางบนโต๊ะที่ตำแหน่งสุ่มรอบจุด spawn (หมุนรอบแกนตั้งได้),
   เต้ารับ+เสาสุ่มตำแหน่ง/มุม
3. `scene_bank` ใช้ feasibility ของ A (IK จับด้านข้าง, `plan_carry`) แทนของ B
4. ทดสอบ v1 ก่อน แล้ว v2 — preflight บนฉากสุ่ม, `--record`, replay แบบเข้มงวด

ความเสี่ยง: จุดเดียวที่ยังไม่รู้คือ controller ของ A ทนตำแหน่งสุ่มได้แค่ไหน — ควรวัดก่อนทำข้ออื่น
(รัน probe บนตำแหน่งปลั๊ก/เต้ารับสุ่ม ~10 ฉาก) ถ้าผ่านน้อย แนวทาง C ก็ต้องจำกัดช่วงสุ่มให้แคบ

**อัปเดต 2026-10-01:** เจ้าของเลือก B (automation ของ automateData บนฉาก setup_env) — v1 10/16, v2 3/3 แล้ว
