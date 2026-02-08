# Transformer Health Assessment Report

## TR-007: Eta Transmission Autotransformer

---

### Executive Summary

| Parameter | Value |
|-----------|-------|
| **Status** | ⚠️ WARNING |
| **Health Score** | 58/100 |
| **Risk Level** | Moderate-High |
| **Scenario** | LTC Degradation + Oil Contamination |

---

### Transformer Information

| Attribute | Details |
|-----------|---------|
| Transformer ID | TR-007 |
| Name | Eta Transmission Autotransformer |
| Location | Northern Transmission Hub |
| Rating | 200 MVA |
| Voltage | 400/220/33 kV |
| Age | 18 years |
| Last Maintenance | 2025-04-18 |
| Next Scheduled | 2026-01-25 |

---

### Current Operating Parameters

#### Temperature Readings - ⚠️ Elevated
- **Top Oil Temperature:** 78°C (Normal: <65°C) ❌ High
- **Winding Temperature:** 92°C (Normal: <90°C) ❌ High
- **Current Load:** 88% of rated capacity

#### Oil Quality Parameters
| Parameter | Value | Limit | Status |
|-----------|-------|-------|--------|
| Moisture Content | 42 ppm | <25 ppm | ❌ High |
| Tan Delta | 1.78% | <1.0% | ❌ High |
| Breakdown Voltage | 32 kV | >50 kV | ❌ Low |

#### Dissolved Gas Analysis (DGA)
| Gas | Concentration | Limit | Status |
|-----|---------------|-------|--------|
| Hydrogen (H₂) | 320 ppm | <100 ppm | ❌ High |
| Methane (CH₄) | 185 ppm | <50 ppm | ❌ High |
| Acetylene (C₂H₂) | 12 ppm | <3 ppm | ❌ Elevated |
| Carbon Monoxide (CO) | 295 ppm | <300 ppm | ⚠️ Near limit |
| Carbon Dioxide (CO₂) | 1520 ppm | <2500 ppm | ✅ Normal |

---

### Assessment Summary

**Diagnosis:** Combined degradation from Load Tap Changer (LTC) issues and oil contamination. Contact wear in the LTC is causing elevated contact resistance and arcing. Carbon particles from LTC operations have contaminated the main tank oil.

### LTC Assessment
- Contact resistance elevated (>25% above baseline)
- Diverter switch showing wear patterns
- Carbon generation from arcing during tap changes
- Oil in LTC compartment severely contaminated

### Root Cause Analysis
1. **Primary:** LTC contact degradation after 18 years
2. **Secondary:** Oil contamination from LTC arcing
3. **Contributing:** High load operation (88%)

### Risk Factors
- LTC contact degradation requiring maintenance
- Oil contamination affecting insulation
- High load operation stressing system
- Carbon particles reducing oil quality

### Recommendations
1. **Within 14 days:** LTC maintenance and contact replacement
2. **Required:** Oil filtration for main tank and LTC compartment
3. **Perform:** Contact resistance testing and timing analysis
4. **Consider:** Load redistribution to reduce stress

---

**Report Generated:** January 12, 2026  
**Classification:** Warning - Maintenance Required Within 14 Days
