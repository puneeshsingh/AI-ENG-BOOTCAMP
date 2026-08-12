# Northwind Robotics — sample RAG corpus

Fictional internal policy pack for TAI / AI Engineering Week 2 (RAG).

## Teaching facts preserved

| Doc ID   | File                         | Must-keep facts |
|----------|------------------------------|-----------------|
| POL-101  | `doc1_handbook.txt`         | Hours 09:00–17:30; remote up to **3 days/week**; full remote needs **director approval** + 6-month review; Slack reachability **10:00–15:00**; annual leave **28 days** + public holidays. Parental leave detail is **intentionally absent** (refusal demo). |
| POL-114  | `doc2_expenses.txt`         | Mileage **45p** over 50 miles; meals **£30/day** overnight; claims within **30 days** with receipt; no receipt → rejected. |
| POL-207  | `doc3_security.txt`         | Password **≥14** chars, rotate **90 days**; **MFA** mandatory; laptops encrypted; lost device reported within **1 hour**. |
| SPEC-WB9 | `doc4_product.txt`          | Payload **25 kg**; battery **8h** / **90 min** charge; max **1.5 m/s**; **0–40 °C**. |
| POL-220  | `doc5_it_acceptable_use.txt`| Extra IT noise for retrieval ranking demos. |
| POL-118  | `doc6_facilities.txt`       | Extra facilities noise for retrieval ranking demos. |

## Download

- Individual files in this folder
- Zip of the whole pack: [`northwind-sample-docs.zip`](./northwind-sample-docs.zip)

## Suggested golden-set questions

1. How many remote days are allowed? → POL-101  
2. What is the mileage rate? → POL-114  
3. How quickly must a lost laptop be reported? → POL-207  
4. What is the WB-9 payload limit? → SPEC-WB9  
5. What is the parental leave policy? → **refuse** (not in corpus)
