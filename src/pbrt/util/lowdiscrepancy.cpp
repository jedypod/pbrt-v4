// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// sampling/lowdiscrepancy.cpp*
#include <pbrt/util/lowdiscrepancy.h>

#include <pbrt/util/bits.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/print.h>
#include <pbrt/util/shuffle.h>
#include <pbrt/util/stats.h>

namespace pbrt {

std::string DigitPermutation::ToString() const {
    std::string s = StringPrintf(
        "[ DigitPermitation base: %d nDigits: %d permutations: ", base, nDigits);
    for (int digitIndex = 0; digitIndex < nDigits; ++digitIndex) {
        s += StringPrintf("[%d] ( ", digitIndex);
        for (int digitValue = 0; digitValue < base; ++digitValue) {
            s += StringPrintf("%d", Perm(digitIndex, digitValue));
            if (digitValue != base - 1)
                s += ", ";
        }
        s += ") ";
    }

    return s + " ]";
}

// HaltonIndexer Local Constants
constexpr int HaltonPixelIndexer::MaxHaltonResolution;

static void extendedGCD(uint64_t a, uint64_t b, int64_t *x, int64_t *y);
static uint64_t multiplicativeInverse(int64_t a, int64_t n) {
    int64_t x, y;
    extendedGCD(a, n, &x, &y);
    return Mod(x, n);
}

static void extendedGCD(uint64_t a, uint64_t b, int64_t *x, int64_t *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return;
    }
    int64_t d = a / b, xp, yp;
    extendedGCD(b, a % b, &xp, &yp);
    *x = yp;
    *y = xp - (d * yp);
}

HaltonPixelIndexer::HaltonPixelIndexer(const Point2i &fullResolution) {
    // Find radical inverse base scales and exponents that cover the image
    for (int i = 0; i < 2; ++i) {
        int base = (i == 0) ? 2 : 3;
        int scale = 1, exp = 0;
        while (scale < std::min(fullResolution[i], MaxHaltonResolution)) {
            scale *= base;
            ++exp;
        }
        baseScales[i] = scale;
        baseExponents[i] = exp;
    }

    // Compute multiplicative inverses for _baseScales_
    multInverse[0] = multiplicativeInverse(baseScales[1], baseScales[0]);
    multInverse[1] = multiplicativeInverse(baseScales[0], baseScales[1]);
}

std::string ToString(RandomizeStrategy r) {
    switch (r) {
    case RandomizeStrategy::None:
        return "None";
    case RandomizeStrategy::CranleyPatterson:
        return "CranleyPatterson";
    case RandomizeStrategy::Xor:
        return "Xor";
    case RandomizeStrategy::Owen:
        return "Owen";
    default:
        LOG_FATAL("Unhandled RandomizeStrategy");
        return "";
    }
}

// Low Discrepancy Static Functions
template <int base>
PBRT_NOINLINE PBRT_CPU_GPU static Float RadicalInverseSpecialized(uint64_t a) {
    const Float invBase = (Float)1 / (Float)base;
    uint64_t reversedDigits = 0;
    Float invBaseN = 1;
    while (a) {
        uint64_t next = a / base;
        uint64_t digit = a - next * base;
        reversedDigits = reversedDigits * base + digit;
        invBaseN *= invBase;
        a = next;
    }
    DCHECK_LT(reversedDigits * invBaseN, 1.00001);
    return std::min(reversedDigits * invBaseN, OneMinusEpsilon);
}

template <int base>
PBRT_NOINLINE PBRT_CPU_GPU static Float ScrambledRadicalInverseSpecialized(
    uint64_t a, const DigitPermutation &perm) {
    CHECK_EQ(perm.base, base);
    const Float invBase = (Float)1 / (Float)base;
    uint64_t reversedDigits = 0;
    Float invBaseN = 1;
    int digitIndex = 0;
    while (1 - invBaseN < 1) {
        uint64_t next = a / base;
        int digitValue = a - next * base;
        reversedDigits = reversedDigits * base + perm.Permute(digitIndex, digitValue);
        invBaseN *= invBase;
        ++digitIndex;
        a = next;
    }
    return std::min(invBaseN * reversedDigits, OneMinusEpsilon);
}

template <int base>
PBRT_NOINLINE PBRT_CPU_GPU static Float ScrambledRadicalInverseSpecialized(
    uint64_t a, uint32_t seed) {
    const Float invBase = (Float)1 / (Float)base;
    uint64_t reversedDigits = 0;
    Float invBaseN = 1;
    int digitIndex = 0;
    while (1 - invBaseN < 1) {
        uint64_t next = a / base;
        uint64_t digit = a - next * base;
        // NOTE: can do Owen scrambling if digitSeed also incorporates
        // reversedDigits...
        uint32_t digitSeed = ((base * 32) + digitIndex) ^ seed ^ reversedDigits;
        reversedDigits =
            reversedDigits * base + PermutationElement(digit, base, digitSeed);
        invBaseN *= invBase;
        ++digitIndex;
        a = next;
    }
    return std::min(invBaseN * reversedDigits, OneMinusEpsilon);
}

// Low Discrepancy Function Definitions
Float RadicalInverse(int baseIndex, uint64_t a) {
    switch (baseIndex) {
    case 0:
        // Compute base-2 radical inverse
#ifndef PBRT_HAVE_HEX_FP_CONSTANTS
        return ReverseBits64(a) * 5.4210108624275222e-20;
#else
        return ReverseBits64(a) * 0x1p-64;
#endif
    case 1:
        return RadicalInverseSpecialized<3>(a);
    case 2:
        return RadicalInverseSpecialized<5>(a);
    case 3:
        return RadicalInverseSpecialized<7>(a);
    // Remainder of cases for _RadicalInverse()_
    case 4:
        return RadicalInverseSpecialized<11>(a);
    case 5:
        return RadicalInverseSpecialized<13>(a);
    case 6:
        return RadicalInverseSpecialized<17>(a);
    case 7:
        return RadicalInverseSpecialized<19>(a);
    case 8:
        return RadicalInverseSpecialized<23>(a);
    case 9:
        return RadicalInverseSpecialized<29>(a);
    case 10:
        return RadicalInverseSpecialized<31>(a);
    case 11:
        return RadicalInverseSpecialized<37>(a);
    case 12:
        return RadicalInverseSpecialized<41>(a);
    case 13:
        return RadicalInverseSpecialized<43>(a);
    case 14:
        return RadicalInverseSpecialized<47>(a);
    case 15:
        return RadicalInverseSpecialized<53>(a);
    case 16:
        return RadicalInverseSpecialized<59>(a);
    case 17:
        return RadicalInverseSpecialized<61>(a);
    case 18:
        return RadicalInverseSpecialized<67>(a);
    case 19:
        return RadicalInverseSpecialized<71>(a);
    case 20:
        return RadicalInverseSpecialized<73>(a);
    case 21:
        return RadicalInverseSpecialized<79>(a);
    case 22:
        return RadicalInverseSpecialized<83>(a);
    case 23:
        return RadicalInverseSpecialized<89>(a);
    case 24:
        return RadicalInverseSpecialized<97>(a);
    case 25:
        return RadicalInverseSpecialized<101>(a);
    case 26:
        return RadicalInverseSpecialized<103>(a);
    case 27:
        return RadicalInverseSpecialized<107>(a);
    case 28:
        return RadicalInverseSpecialized<109>(a);
    case 29:
        return RadicalInverseSpecialized<113>(a);
    case 30:
        return RadicalInverseSpecialized<127>(a);
    case 31:
        return RadicalInverseSpecialized<131>(a);
    case 32:
        return RadicalInverseSpecialized<137>(a);
    case 33:
        return RadicalInverseSpecialized<139>(a);
    case 34:
        return RadicalInverseSpecialized<149>(a);
    case 35:
        return RadicalInverseSpecialized<151>(a);
    case 36:
        return RadicalInverseSpecialized<157>(a);
    case 37:
        return RadicalInverseSpecialized<163>(a);
    case 38:
        return RadicalInverseSpecialized<167>(a);
    case 39:
        return RadicalInverseSpecialized<173>(a);
    case 40:
        return RadicalInverseSpecialized<179>(a);
    case 41:
        return RadicalInverseSpecialized<181>(a);
    case 42:
        return RadicalInverseSpecialized<191>(a);
    case 43:
        return RadicalInverseSpecialized<193>(a);
    case 44:
        return RadicalInverseSpecialized<197>(a);
    case 45:
        return RadicalInverseSpecialized<199>(a);
    case 46:
        return RadicalInverseSpecialized<211>(a);
    case 47:
        return RadicalInverseSpecialized<223>(a);
    case 48:
        return RadicalInverseSpecialized<227>(a);
    case 49:
        return RadicalInverseSpecialized<229>(a);
    case 50:
        return RadicalInverseSpecialized<233>(a);
    case 51:
        return RadicalInverseSpecialized<239>(a);
    case 52:
        return RadicalInverseSpecialized<241>(a);
    case 53:
        return RadicalInverseSpecialized<251>(a);
    case 54:
        return RadicalInverseSpecialized<257>(a);
    case 55:
        return RadicalInverseSpecialized<263>(a);
    case 56:
        return RadicalInverseSpecialized<269>(a);
    case 57:
        return RadicalInverseSpecialized<271>(a);
    case 58:
        return RadicalInverseSpecialized<277>(a);
    case 59:
        return RadicalInverseSpecialized<281>(a);
    case 60:
        return RadicalInverseSpecialized<283>(a);
    case 61:
        return RadicalInverseSpecialized<293>(a);
    case 62:
        return RadicalInverseSpecialized<307>(a);
    case 63:
        return RadicalInverseSpecialized<311>(a);
    case 64:
        return RadicalInverseSpecialized<313>(a);
    case 65:
        return RadicalInverseSpecialized<317>(a);
    case 66:
        return RadicalInverseSpecialized<331>(a);
    case 67:
        return RadicalInverseSpecialized<337>(a);
    case 68:
        return RadicalInverseSpecialized<347>(a);
    case 69:
        return RadicalInverseSpecialized<349>(a);
    case 70:
        return RadicalInverseSpecialized<353>(a);
    case 71:
        return RadicalInverseSpecialized<359>(a);
    case 72:
        return RadicalInverseSpecialized<367>(a);
    case 73:
        return RadicalInverseSpecialized<373>(a);
    case 74:
        return RadicalInverseSpecialized<379>(a);
    case 75:
        return RadicalInverseSpecialized<383>(a);
    case 76:
        return RadicalInverseSpecialized<389>(a);
    case 77:
        return RadicalInverseSpecialized<397>(a);
    case 78:
        return RadicalInverseSpecialized<401>(a);
    case 79:
        return RadicalInverseSpecialized<409>(a);
    case 80:
        return RadicalInverseSpecialized<419>(a);
    case 81:
        return RadicalInverseSpecialized<421>(a);
    case 82:
        return RadicalInverseSpecialized<431>(a);
    case 83:
        return RadicalInverseSpecialized<433>(a);
    case 84:
        return RadicalInverseSpecialized<439>(a);
    case 85:
        return RadicalInverseSpecialized<443>(a);
    case 86:
        return RadicalInverseSpecialized<449>(a);
    case 87:
        return RadicalInverseSpecialized<457>(a);
    case 88:
        return RadicalInverseSpecialized<461>(a);
    case 89:
        return RadicalInverseSpecialized<463>(a);
    case 90:
        return RadicalInverseSpecialized<467>(a);
    case 91:
        return RadicalInverseSpecialized<479>(a);
    case 92:
        return RadicalInverseSpecialized<487>(a);
    case 93:
        return RadicalInverseSpecialized<491>(a);
    case 94:
        return RadicalInverseSpecialized<499>(a);
    case 95:
        return RadicalInverseSpecialized<503>(a);
    case 96:
        return RadicalInverseSpecialized<509>(a);
    case 97:
        return RadicalInverseSpecialized<521>(a);
    case 98:
        return RadicalInverseSpecialized<523>(a);
    case 99:
        return RadicalInverseSpecialized<541>(a);
    case 100:
        return RadicalInverseSpecialized<547>(a);
    case 101:
        return RadicalInverseSpecialized<557>(a);
    case 102:
        return RadicalInverseSpecialized<563>(a);
    case 103:
        return RadicalInverseSpecialized<569>(a);
    case 104:
        return RadicalInverseSpecialized<571>(a);
    case 105:
        return RadicalInverseSpecialized<577>(a);
    case 106:
        return RadicalInverseSpecialized<587>(a);
    case 107:
        return RadicalInverseSpecialized<593>(a);
    case 108:
        return RadicalInverseSpecialized<599>(a);
    case 109:
        return RadicalInverseSpecialized<601>(a);
    case 110:
        return RadicalInverseSpecialized<607>(a);
    case 111:
        return RadicalInverseSpecialized<613>(a);
    case 112:
        return RadicalInverseSpecialized<617>(a);
    case 113:
        return RadicalInverseSpecialized<619>(a);
    case 114:
        return RadicalInverseSpecialized<631>(a);
    case 115:
        return RadicalInverseSpecialized<641>(a);
    case 116:
        return RadicalInverseSpecialized<643>(a);
    case 117:
        return RadicalInverseSpecialized<647>(a);
    case 118:
        return RadicalInverseSpecialized<653>(a);
    case 119:
        return RadicalInverseSpecialized<659>(a);
    case 120:
        return RadicalInverseSpecialized<661>(a);
    case 121:
        return RadicalInverseSpecialized<673>(a);
    case 122:
        return RadicalInverseSpecialized<677>(a);
    case 123:
        return RadicalInverseSpecialized<683>(a);
    case 124:
        return RadicalInverseSpecialized<691>(a);
    case 125:
        return RadicalInverseSpecialized<701>(a);
    case 126:
        return RadicalInverseSpecialized<709>(a);
    case 127:
        return RadicalInverseSpecialized<719>(a);
    case 128:
        return RadicalInverseSpecialized<727>(a);
    case 129:
        return RadicalInverseSpecialized<733>(a);
    case 130:
        return RadicalInverseSpecialized<739>(a);
    case 131:
        return RadicalInverseSpecialized<743>(a);
    case 132:
        return RadicalInverseSpecialized<751>(a);
    case 133:
        return RadicalInverseSpecialized<757>(a);
    case 134:
        return RadicalInverseSpecialized<761>(a);
    case 135:
        return RadicalInverseSpecialized<769>(a);
    case 136:
        return RadicalInverseSpecialized<773>(a);
    case 137:
        return RadicalInverseSpecialized<787>(a);
    case 138:
        return RadicalInverseSpecialized<797>(a);
    case 139:
        return RadicalInverseSpecialized<809>(a);
    case 140:
        return RadicalInverseSpecialized<811>(a);
    case 141:
        return RadicalInverseSpecialized<821>(a);
    case 142:
        return RadicalInverseSpecialized<823>(a);
    case 143:
        return RadicalInverseSpecialized<827>(a);
    case 144:
        return RadicalInverseSpecialized<829>(a);
    case 145:
        return RadicalInverseSpecialized<839>(a);
    case 146:
        return RadicalInverseSpecialized<853>(a);
    case 147:
        return RadicalInverseSpecialized<857>(a);
    case 148:
        return RadicalInverseSpecialized<859>(a);
    case 149:
        return RadicalInverseSpecialized<863>(a);
    case 150:
        return RadicalInverseSpecialized<877>(a);
    case 151:
        return RadicalInverseSpecialized<881>(a);
    case 152:
        return RadicalInverseSpecialized<883>(a);
    case 153:
        return RadicalInverseSpecialized<887>(a);
    case 154:
        return RadicalInverseSpecialized<907>(a);
    case 155:
        return RadicalInverseSpecialized<911>(a);
    case 156:
        return RadicalInverseSpecialized<919>(a);
    case 157:
        return RadicalInverseSpecialized<929>(a);
    case 158:
        return RadicalInverseSpecialized<937>(a);
    case 159:
        return RadicalInverseSpecialized<941>(a);
    case 160:
        return RadicalInverseSpecialized<947>(a);
    case 161:
        return RadicalInverseSpecialized<953>(a);
    case 162:
        return RadicalInverseSpecialized<967>(a);
    case 163:
        return RadicalInverseSpecialized<971>(a);
    case 164:
        return RadicalInverseSpecialized<977>(a);
    case 165:
        return RadicalInverseSpecialized<983>(a);
    case 166:
        return RadicalInverseSpecialized<991>(a);
    case 167:
        return RadicalInverseSpecialized<997>(a);
    case 168:
        return RadicalInverseSpecialized<1009>(a);
    case 169:
        return RadicalInverseSpecialized<1013>(a);
    case 170:
        return RadicalInverseSpecialized<1019>(a);
    case 171:
        return RadicalInverseSpecialized<1021>(a);
    case 172:
        return RadicalInverseSpecialized<1031>(a);
    case 173:
        return RadicalInverseSpecialized<1033>(a);
    case 174:
        return RadicalInverseSpecialized<1039>(a);
    case 175:
        return RadicalInverseSpecialized<1049>(a);
    case 176:
        return RadicalInverseSpecialized<1051>(a);
    case 177:
        return RadicalInverseSpecialized<1061>(a);
    case 178:
        return RadicalInverseSpecialized<1063>(a);
    case 179:
        return RadicalInverseSpecialized<1069>(a);
    case 180:
        return RadicalInverseSpecialized<1087>(a);
    case 181:
        return RadicalInverseSpecialized<1091>(a);
    case 182:
        return RadicalInverseSpecialized<1093>(a);
    case 183:
        return RadicalInverseSpecialized<1097>(a);
    case 184:
        return RadicalInverseSpecialized<1103>(a);
    case 185:
        return RadicalInverseSpecialized<1109>(a);
    case 186:
        return RadicalInverseSpecialized<1117>(a);
    case 187:
        return RadicalInverseSpecialized<1123>(a);
    case 188:
        return RadicalInverseSpecialized<1129>(a);
    case 189:
        return RadicalInverseSpecialized<1151>(a);
    case 190:
        return RadicalInverseSpecialized<1153>(a);
    case 191:
        return RadicalInverseSpecialized<1163>(a);
    case 192:
        return RadicalInverseSpecialized<1171>(a);
    case 193:
        return RadicalInverseSpecialized<1181>(a);
    case 194:
        return RadicalInverseSpecialized<1187>(a);
    case 195:
        return RadicalInverseSpecialized<1193>(a);
    case 196:
        return RadicalInverseSpecialized<1201>(a);
    case 197:
        return RadicalInverseSpecialized<1213>(a);
    case 198:
        return RadicalInverseSpecialized<1217>(a);
    case 199:
        return RadicalInverseSpecialized<1223>(a);
    case 200:
        return RadicalInverseSpecialized<1229>(a);
    case 201:
        return RadicalInverseSpecialized<1231>(a);
    case 202:
        return RadicalInverseSpecialized<1237>(a);
    case 203:
        return RadicalInverseSpecialized<1249>(a);
    case 204:
        return RadicalInverseSpecialized<1259>(a);
    case 205:
        return RadicalInverseSpecialized<1277>(a);
    case 206:
        return RadicalInverseSpecialized<1279>(a);
    case 207:
        return RadicalInverseSpecialized<1283>(a);
    case 208:
        return RadicalInverseSpecialized<1289>(a);
    case 209:
        return RadicalInverseSpecialized<1291>(a);
    case 210:
        return RadicalInverseSpecialized<1297>(a);
    case 211:
        return RadicalInverseSpecialized<1301>(a);
    case 212:
        return RadicalInverseSpecialized<1303>(a);
    case 213:
        return RadicalInverseSpecialized<1307>(a);
    case 214:
        return RadicalInverseSpecialized<1319>(a);
    case 215:
        return RadicalInverseSpecialized<1321>(a);
    case 216:
        return RadicalInverseSpecialized<1327>(a);
    case 217:
        return RadicalInverseSpecialized<1361>(a);
    case 218:
        return RadicalInverseSpecialized<1367>(a);
    case 219:
        return RadicalInverseSpecialized<1373>(a);
    case 220:
        return RadicalInverseSpecialized<1381>(a);
    case 221:
        return RadicalInverseSpecialized<1399>(a);
    case 222:
        return RadicalInverseSpecialized<1409>(a);
    case 223:
        return RadicalInverseSpecialized<1423>(a);
    case 224:
        return RadicalInverseSpecialized<1427>(a);
    case 225:
        return RadicalInverseSpecialized<1429>(a);
    case 226:
        return RadicalInverseSpecialized<1433>(a);
    case 227:
        return RadicalInverseSpecialized<1439>(a);
    case 228:
        return RadicalInverseSpecialized<1447>(a);
    case 229:
        return RadicalInverseSpecialized<1451>(a);
    case 230:
        return RadicalInverseSpecialized<1453>(a);
    case 231:
        return RadicalInverseSpecialized<1459>(a);
    case 232:
        return RadicalInverseSpecialized<1471>(a);
    case 233:
        return RadicalInverseSpecialized<1481>(a);
    case 234:
        return RadicalInverseSpecialized<1483>(a);
    case 235:
        return RadicalInverseSpecialized<1487>(a);
    case 236:
        return RadicalInverseSpecialized<1489>(a);
    case 237:
        return RadicalInverseSpecialized<1493>(a);
    case 238:
        return RadicalInverseSpecialized<1499>(a);
    case 239:
        return RadicalInverseSpecialized<1511>(a);
    case 240:
        return RadicalInverseSpecialized<1523>(a);
    case 241:
        return RadicalInverseSpecialized<1531>(a);
    case 242:
        return RadicalInverseSpecialized<1543>(a);
    case 243:
        return RadicalInverseSpecialized<1549>(a);
    case 244:
        return RadicalInverseSpecialized<1553>(a);
    case 245:
        return RadicalInverseSpecialized<1559>(a);
    case 246:
        return RadicalInverseSpecialized<1567>(a);
    case 247:
        return RadicalInverseSpecialized<1571>(a);
    case 248:
        return RadicalInverseSpecialized<1579>(a);
    case 249:
        return RadicalInverseSpecialized<1583>(a);
    case 250:
        return RadicalInverseSpecialized<1597>(a);
    case 251:
        return RadicalInverseSpecialized<1601>(a);
    case 252:
        return RadicalInverseSpecialized<1607>(a);
    case 253:
        return RadicalInverseSpecialized<1609>(a);
    case 254:
        return RadicalInverseSpecialized<1613>(a);
    case 255:
        return RadicalInverseSpecialized<1619>(a);
    case 256:
        return RadicalInverseSpecialized<1621>(a);
    case 257:
        return RadicalInverseSpecialized<1627>(a);
    case 258:
        return RadicalInverseSpecialized<1637>(a);
    case 259:
        return RadicalInverseSpecialized<1657>(a);
    case 260:
        return RadicalInverseSpecialized<1663>(a);
    case 261:
        return RadicalInverseSpecialized<1667>(a);
    case 262:
        return RadicalInverseSpecialized<1669>(a);
    case 263:
        return RadicalInverseSpecialized<1693>(a);
    case 264:
        return RadicalInverseSpecialized<1697>(a);
    case 265:
        return RadicalInverseSpecialized<1699>(a);
    case 266:
        return RadicalInverseSpecialized<1709>(a);
    case 267:
        return RadicalInverseSpecialized<1721>(a);
    case 268:
        return RadicalInverseSpecialized<1723>(a);
    case 269:
        return RadicalInverseSpecialized<1733>(a);
    case 270:
        return RadicalInverseSpecialized<1741>(a);
    case 271:
        return RadicalInverseSpecialized<1747>(a);
    case 272:
        return RadicalInverseSpecialized<1753>(a);
    case 273:
        return RadicalInverseSpecialized<1759>(a);
    case 274:
        return RadicalInverseSpecialized<1777>(a);
    case 275:
        return RadicalInverseSpecialized<1783>(a);
    case 276:
        return RadicalInverseSpecialized<1787>(a);
    case 277:
        return RadicalInverseSpecialized<1789>(a);
    case 278:
        return RadicalInverseSpecialized<1801>(a);
    case 279:
        return RadicalInverseSpecialized<1811>(a);
    case 280:
        return RadicalInverseSpecialized<1823>(a);
    case 281:
        return RadicalInverseSpecialized<1831>(a);
    case 282:
        return RadicalInverseSpecialized<1847>(a);
    case 283:
        return RadicalInverseSpecialized<1861>(a);
    case 284:
        return RadicalInverseSpecialized<1867>(a);
    case 285:
        return RadicalInverseSpecialized<1871>(a);
    case 286:
        return RadicalInverseSpecialized<1873>(a);
    case 287:
        return RadicalInverseSpecialized<1877>(a);
    case 288:
        return RadicalInverseSpecialized<1879>(a);
    case 289:
        return RadicalInverseSpecialized<1889>(a);
    case 290:
        return RadicalInverseSpecialized<1901>(a);
    case 291:
        return RadicalInverseSpecialized<1907>(a);
    case 292:
        return RadicalInverseSpecialized<1913>(a);
    case 293:
        return RadicalInverseSpecialized<1931>(a);
    case 294:
        return RadicalInverseSpecialized<1933>(a);
    case 295:
        return RadicalInverseSpecialized<1949>(a);
    case 296:
        return RadicalInverseSpecialized<1951>(a);
    case 297:
        return RadicalInverseSpecialized<1973>(a);
    case 298:
        return RadicalInverseSpecialized<1979>(a);
    case 299:
        return RadicalInverseSpecialized<1987>(a);
    case 300:
        return RadicalInverseSpecialized<1993>(a);
    case 301:
        return RadicalInverseSpecialized<1997>(a);
    case 302:
        return RadicalInverseSpecialized<1999>(a);
    case 303:
        return RadicalInverseSpecialized<2003>(a);
    case 304:
        return RadicalInverseSpecialized<2011>(a);
    case 305:
        return RadicalInverseSpecialized<2017>(a);
    case 306:
        return RadicalInverseSpecialized<2027>(a);
    case 307:
        return RadicalInverseSpecialized<2029>(a);
    case 308:
        return RadicalInverseSpecialized<2039>(a);
    case 309:
        return RadicalInverseSpecialized<2053>(a);
    case 310:
        return RadicalInverseSpecialized<2063>(a);
    case 311:
        return RadicalInverseSpecialized<2069>(a);
    case 312:
        return RadicalInverseSpecialized<2081>(a);
    case 313:
        return RadicalInverseSpecialized<2083>(a);
    case 314:
        return RadicalInverseSpecialized<2087>(a);
    case 315:
        return RadicalInverseSpecialized<2089>(a);
    case 316:
        return RadicalInverseSpecialized<2099>(a);
    case 317:
        return RadicalInverseSpecialized<2111>(a);
    case 318:
        return RadicalInverseSpecialized<2113>(a);
    case 319:
        return RadicalInverseSpecialized<2129>(a);
    case 320:
        return RadicalInverseSpecialized<2131>(a);
    case 321:
        return RadicalInverseSpecialized<2137>(a);
    case 322:
        return RadicalInverseSpecialized<2141>(a);
    case 323:
        return RadicalInverseSpecialized<2143>(a);
    case 324:
        return RadicalInverseSpecialized<2153>(a);
    case 325:
        return RadicalInverseSpecialized<2161>(a);
    case 326:
        return RadicalInverseSpecialized<2179>(a);
    case 327:
        return RadicalInverseSpecialized<2203>(a);
    case 328:
        return RadicalInverseSpecialized<2207>(a);
    case 329:
        return RadicalInverseSpecialized<2213>(a);
    case 330:
        return RadicalInverseSpecialized<2221>(a);
    case 331:
        return RadicalInverseSpecialized<2237>(a);
    case 332:
        return RadicalInverseSpecialized<2239>(a);
    case 333:
        return RadicalInverseSpecialized<2243>(a);
    case 334:
        return RadicalInverseSpecialized<2251>(a);
    case 335:
        return RadicalInverseSpecialized<2267>(a);
    case 336:
        return RadicalInverseSpecialized<2269>(a);
    case 337:
        return RadicalInverseSpecialized<2273>(a);
    case 338:
        return RadicalInverseSpecialized<2281>(a);
    case 339:
        return RadicalInverseSpecialized<2287>(a);
    case 340:
        return RadicalInverseSpecialized<2293>(a);
    case 341:
        return RadicalInverseSpecialized<2297>(a);
    case 342:
        return RadicalInverseSpecialized<2309>(a);
    case 343:
        return RadicalInverseSpecialized<2311>(a);
    case 344:
        return RadicalInverseSpecialized<2333>(a);
    case 345:
        return RadicalInverseSpecialized<2339>(a);
    case 346:
        return RadicalInverseSpecialized<2341>(a);
    case 347:
        return RadicalInverseSpecialized<2347>(a);
    case 348:
        return RadicalInverseSpecialized<2351>(a);
    case 349:
        return RadicalInverseSpecialized<2357>(a);
    case 350:
        return RadicalInverseSpecialized<2371>(a);
    case 351:
        return RadicalInverseSpecialized<2377>(a);
    case 352:
        return RadicalInverseSpecialized<2381>(a);
    case 353:
        return RadicalInverseSpecialized<2383>(a);
    case 354:
        return RadicalInverseSpecialized<2389>(a);
    case 355:
        return RadicalInverseSpecialized<2393>(a);
    case 356:
        return RadicalInverseSpecialized<2399>(a);
    case 357:
        return RadicalInverseSpecialized<2411>(a);
    case 358:
        return RadicalInverseSpecialized<2417>(a);
    case 359:
        return RadicalInverseSpecialized<2423>(a);
    case 360:
        return RadicalInverseSpecialized<2437>(a);
    case 361:
        return RadicalInverseSpecialized<2441>(a);
    case 362:
        return RadicalInverseSpecialized<2447>(a);
    case 363:
        return RadicalInverseSpecialized<2459>(a);
    case 364:
        return RadicalInverseSpecialized<2467>(a);
    case 365:
        return RadicalInverseSpecialized<2473>(a);
    case 366:
        return RadicalInverseSpecialized<2477>(a);
    case 367:
        return RadicalInverseSpecialized<2503>(a);
    case 368:
        return RadicalInverseSpecialized<2521>(a);
    case 369:
        return RadicalInverseSpecialized<2531>(a);
    case 370:
        return RadicalInverseSpecialized<2539>(a);
    case 371:
        return RadicalInverseSpecialized<2543>(a);
    case 372:
        return RadicalInverseSpecialized<2549>(a);
    case 373:
        return RadicalInverseSpecialized<2551>(a);
    case 374:
        return RadicalInverseSpecialized<2557>(a);
    case 375:
        return RadicalInverseSpecialized<2579>(a);
    case 376:
        return RadicalInverseSpecialized<2591>(a);
    case 377:
        return RadicalInverseSpecialized<2593>(a);
    case 378:
        return RadicalInverseSpecialized<2609>(a);
    case 379:
        return RadicalInverseSpecialized<2617>(a);
    case 380:
        return RadicalInverseSpecialized<2621>(a);
    case 381:
        return RadicalInverseSpecialized<2633>(a);
    case 382:
        return RadicalInverseSpecialized<2647>(a);
    case 383:
        return RadicalInverseSpecialized<2657>(a);
    case 384:
        return RadicalInverseSpecialized<2659>(a);
    case 385:
        return RadicalInverseSpecialized<2663>(a);
    case 386:
        return RadicalInverseSpecialized<2671>(a);
    case 387:
        return RadicalInverseSpecialized<2677>(a);
    case 388:
        return RadicalInverseSpecialized<2683>(a);
    case 389:
        return RadicalInverseSpecialized<2687>(a);
    case 390:
        return RadicalInverseSpecialized<2689>(a);
    case 391:
        return RadicalInverseSpecialized<2693>(a);
    case 392:
        return RadicalInverseSpecialized<2699>(a);
    case 393:
        return RadicalInverseSpecialized<2707>(a);
    case 394:
        return RadicalInverseSpecialized<2711>(a);
    case 395:
        return RadicalInverseSpecialized<2713>(a);
    case 396:
        return RadicalInverseSpecialized<2719>(a);
    case 397:
        return RadicalInverseSpecialized<2729>(a);
    case 398:
        return RadicalInverseSpecialized<2731>(a);
    case 399:
        return RadicalInverseSpecialized<2741>(a);
    case 400:
        return RadicalInverseSpecialized<2749>(a);
    case 401:
        return RadicalInverseSpecialized<2753>(a);
    case 402:
        return RadicalInverseSpecialized<2767>(a);
    case 403:
        return RadicalInverseSpecialized<2777>(a);
    case 404:
        return RadicalInverseSpecialized<2789>(a);
    case 405:
        return RadicalInverseSpecialized<2791>(a);
    case 406:
        return RadicalInverseSpecialized<2797>(a);
    case 407:
        return RadicalInverseSpecialized<2801>(a);
    case 408:
        return RadicalInverseSpecialized<2803>(a);
    case 409:
        return RadicalInverseSpecialized<2819>(a);
    case 410:
        return RadicalInverseSpecialized<2833>(a);
    case 411:
        return RadicalInverseSpecialized<2837>(a);
    case 412:
        return RadicalInverseSpecialized<2843>(a);
    case 413:
        return RadicalInverseSpecialized<2851>(a);
    case 414:
        return RadicalInverseSpecialized<2857>(a);
    case 415:
        return RadicalInverseSpecialized<2861>(a);
    case 416:
        return RadicalInverseSpecialized<2879>(a);
    case 417:
        return RadicalInverseSpecialized<2887>(a);
    case 418:
        return RadicalInverseSpecialized<2897>(a);
    case 419:
        return RadicalInverseSpecialized<2903>(a);
    case 420:
        return RadicalInverseSpecialized<2909>(a);
    case 421:
        return RadicalInverseSpecialized<2917>(a);
    case 422:
        return RadicalInverseSpecialized<2927>(a);
    case 423:
        return RadicalInverseSpecialized<2939>(a);
    case 424:
        return RadicalInverseSpecialized<2953>(a);
    case 425:
        return RadicalInverseSpecialized<2957>(a);
    case 426:
        return RadicalInverseSpecialized<2963>(a);
    case 427:
        return RadicalInverseSpecialized<2969>(a);
    case 428:
        return RadicalInverseSpecialized<2971>(a);
    case 429:
        return RadicalInverseSpecialized<2999>(a);
    case 430:
        return RadicalInverseSpecialized<3001>(a);
    case 431:
        return RadicalInverseSpecialized<3011>(a);
    case 432:
        return RadicalInverseSpecialized<3019>(a);
    case 433:
        return RadicalInverseSpecialized<3023>(a);
    case 434:
        return RadicalInverseSpecialized<3037>(a);
    case 435:
        return RadicalInverseSpecialized<3041>(a);
    case 436:
        return RadicalInverseSpecialized<3049>(a);
    case 437:
        return RadicalInverseSpecialized<3061>(a);
    case 438:
        return RadicalInverseSpecialized<3067>(a);
    case 439:
        return RadicalInverseSpecialized<3079>(a);
    case 440:
        return RadicalInverseSpecialized<3083>(a);
    case 441:
        return RadicalInverseSpecialized<3089>(a);
    case 442:
        return RadicalInverseSpecialized<3109>(a);
    case 443:
        return RadicalInverseSpecialized<3119>(a);
    case 444:
        return RadicalInverseSpecialized<3121>(a);
    case 445:
        return RadicalInverseSpecialized<3137>(a);
    case 446:
        return RadicalInverseSpecialized<3163>(a);
    case 447:
        return RadicalInverseSpecialized<3167>(a);
    case 448:
        return RadicalInverseSpecialized<3169>(a);
    case 449:
        return RadicalInverseSpecialized<3181>(a);
    case 450:
        return RadicalInverseSpecialized<3187>(a);
    case 451:
        return RadicalInverseSpecialized<3191>(a);
    case 452:
        return RadicalInverseSpecialized<3203>(a);
    case 453:
        return RadicalInverseSpecialized<3209>(a);
    case 454:
        return RadicalInverseSpecialized<3217>(a);
    case 455:
        return RadicalInverseSpecialized<3221>(a);
    case 456:
        return RadicalInverseSpecialized<3229>(a);
    case 457:
        return RadicalInverseSpecialized<3251>(a);
    case 458:
        return RadicalInverseSpecialized<3253>(a);
    case 459:
        return RadicalInverseSpecialized<3257>(a);
    case 460:
        return RadicalInverseSpecialized<3259>(a);
    case 461:
        return RadicalInverseSpecialized<3271>(a);
    case 462:
        return RadicalInverseSpecialized<3299>(a);
    case 463:
        return RadicalInverseSpecialized<3301>(a);
    case 464:
        return RadicalInverseSpecialized<3307>(a);
    case 465:
        return RadicalInverseSpecialized<3313>(a);
    case 466:
        return RadicalInverseSpecialized<3319>(a);
    case 467:
        return RadicalInverseSpecialized<3323>(a);
    case 468:
        return RadicalInverseSpecialized<3329>(a);
    case 469:
        return RadicalInverseSpecialized<3331>(a);
    case 470:
        return RadicalInverseSpecialized<3343>(a);
    case 471:
        return RadicalInverseSpecialized<3347>(a);
    case 472:
        return RadicalInverseSpecialized<3359>(a);
    case 473:
        return RadicalInverseSpecialized<3361>(a);
    case 474:
        return RadicalInverseSpecialized<3371>(a);
    case 475:
        return RadicalInverseSpecialized<3373>(a);
    case 476:
        return RadicalInverseSpecialized<3389>(a);
    case 477:
        return RadicalInverseSpecialized<3391>(a);
    case 478:
        return RadicalInverseSpecialized<3407>(a);
    case 479:
        return RadicalInverseSpecialized<3413>(a);
    case 480:
        return RadicalInverseSpecialized<3433>(a);
    case 481:
        return RadicalInverseSpecialized<3449>(a);
    case 482:
        return RadicalInverseSpecialized<3457>(a);
    case 483:
        return RadicalInverseSpecialized<3461>(a);
    case 484:
        return RadicalInverseSpecialized<3463>(a);
    case 485:
        return RadicalInverseSpecialized<3467>(a);
    case 486:
        return RadicalInverseSpecialized<3469>(a);
    case 487:
        return RadicalInverseSpecialized<3491>(a);
    case 488:
        return RadicalInverseSpecialized<3499>(a);
    case 489:
        return RadicalInverseSpecialized<3511>(a);
    case 490:
        return RadicalInverseSpecialized<3517>(a);
    case 491:
        return RadicalInverseSpecialized<3527>(a);
    case 492:
        return RadicalInverseSpecialized<3529>(a);
    case 493:
        return RadicalInverseSpecialized<3533>(a);
    case 494:
        return RadicalInverseSpecialized<3539>(a);
    case 495:
        return RadicalInverseSpecialized<3541>(a);
    case 496:
        return RadicalInverseSpecialized<3547>(a);
    case 497:
        return RadicalInverseSpecialized<3557>(a);
    case 498:
        return RadicalInverseSpecialized<3559>(a);
    case 499:
        return RadicalInverseSpecialized<3571>(a);
    case 500:
        return RadicalInverseSpecialized<3581>(a);
    case 501:
        return RadicalInverseSpecialized<3583>(a);
    case 502:
        return RadicalInverseSpecialized<3593>(a);
    case 503:
        return RadicalInverseSpecialized<3607>(a);
    case 504:
        return RadicalInverseSpecialized<3613>(a);
    case 505:
        return RadicalInverseSpecialized<3617>(a);
    case 506:
        return RadicalInverseSpecialized<3623>(a);
    case 507:
        return RadicalInverseSpecialized<3631>(a);
    case 508:
        return RadicalInverseSpecialized<3637>(a);
    case 509:
        return RadicalInverseSpecialized<3643>(a);
    case 510:
        return RadicalInverseSpecialized<3659>(a);
    case 511:
        return RadicalInverseSpecialized<3671>(a);
    case 512:
        return RadicalInverseSpecialized<3673>(a);
    case 513:
        return RadicalInverseSpecialized<3677>(a);
    case 514:
        return RadicalInverseSpecialized<3691>(a);
    case 515:
        return RadicalInverseSpecialized<3697>(a);
    case 516:
        return RadicalInverseSpecialized<3701>(a);
    case 517:
        return RadicalInverseSpecialized<3709>(a);
    case 518:
        return RadicalInverseSpecialized<3719>(a);
    case 519:
        return RadicalInverseSpecialized<3727>(a);
    case 520:
        return RadicalInverseSpecialized<3733>(a);
    case 521:
        return RadicalInverseSpecialized<3739>(a);
    case 522:
        return RadicalInverseSpecialized<3761>(a);
    case 523:
        return RadicalInverseSpecialized<3767>(a);
    case 524:
        return RadicalInverseSpecialized<3769>(a);
    case 525:
        return RadicalInverseSpecialized<3779>(a);
    case 526:
        return RadicalInverseSpecialized<3793>(a);
    case 527:
        return RadicalInverseSpecialized<3797>(a);
    case 528:
        return RadicalInverseSpecialized<3803>(a);
    case 529:
        return RadicalInverseSpecialized<3821>(a);
    case 530:
        return RadicalInverseSpecialized<3823>(a);
    case 531:
        return RadicalInverseSpecialized<3833>(a);
    case 532:
        return RadicalInverseSpecialized<3847>(a);
    case 533:
        return RadicalInverseSpecialized<3851>(a);
    case 534:
        return RadicalInverseSpecialized<3853>(a);
    case 535:
        return RadicalInverseSpecialized<3863>(a);
    case 536:
        return RadicalInverseSpecialized<3877>(a);
    case 537:
        return RadicalInverseSpecialized<3881>(a);
    case 538:
        return RadicalInverseSpecialized<3889>(a);
    case 539:
        return RadicalInverseSpecialized<3907>(a);
    case 540:
        return RadicalInverseSpecialized<3911>(a);
    case 541:
        return RadicalInverseSpecialized<3917>(a);
    case 542:
        return RadicalInverseSpecialized<3919>(a);
    case 543:
        return RadicalInverseSpecialized<3923>(a);
    case 544:
        return RadicalInverseSpecialized<3929>(a);
    case 545:
        return RadicalInverseSpecialized<3931>(a);
    case 546:
        return RadicalInverseSpecialized<3943>(a);
    case 547:
        return RadicalInverseSpecialized<3947>(a);
    case 548:
        return RadicalInverseSpecialized<3967>(a);
    case 549:
        return RadicalInverseSpecialized<3989>(a);
    case 550:
        return RadicalInverseSpecialized<4001>(a);
    case 551:
        return RadicalInverseSpecialized<4003>(a);
    case 552:
        return RadicalInverseSpecialized<4007>(a);
    case 553:
        return RadicalInverseSpecialized<4013>(a);
    case 554:
        return RadicalInverseSpecialized<4019>(a);
    case 555:
        return RadicalInverseSpecialized<4021>(a);
    case 556:
        return RadicalInverseSpecialized<4027>(a);
    case 557:
        return RadicalInverseSpecialized<4049>(a);
    case 558:
        return RadicalInverseSpecialized<4051>(a);
    case 559:
        return RadicalInverseSpecialized<4057>(a);
    case 560:
        return RadicalInverseSpecialized<4073>(a);
    case 561:
        return RadicalInverseSpecialized<4079>(a);
    case 562:
        return RadicalInverseSpecialized<4091>(a);
    case 563:
        return RadicalInverseSpecialized<4093>(a);
    case 564:
        return RadicalInverseSpecialized<4099>(a);
    case 565:
        return RadicalInverseSpecialized<4111>(a);
    case 566:
        return RadicalInverseSpecialized<4127>(a);
    case 567:
        return RadicalInverseSpecialized<4129>(a);
    case 568:
        return RadicalInverseSpecialized<4133>(a);
    case 569:
        return RadicalInverseSpecialized<4139>(a);
    case 570:
        return RadicalInverseSpecialized<4153>(a);
    case 571:
        return RadicalInverseSpecialized<4157>(a);
    case 572:
        return RadicalInverseSpecialized<4159>(a);
    case 573:
        return RadicalInverseSpecialized<4177>(a);
    case 574:
        return RadicalInverseSpecialized<4201>(a);
    case 575:
        return RadicalInverseSpecialized<4211>(a);
    case 576:
        return RadicalInverseSpecialized<4217>(a);
    case 577:
        return RadicalInverseSpecialized<4219>(a);
    case 578:
        return RadicalInverseSpecialized<4229>(a);
    case 579:
        return RadicalInverseSpecialized<4231>(a);
    case 580:
        return RadicalInverseSpecialized<4241>(a);
    case 581:
        return RadicalInverseSpecialized<4243>(a);
    case 582:
        return RadicalInverseSpecialized<4253>(a);
    case 583:
        return RadicalInverseSpecialized<4259>(a);
    case 584:
        return RadicalInverseSpecialized<4261>(a);
    case 585:
        return RadicalInverseSpecialized<4271>(a);
    case 586:
        return RadicalInverseSpecialized<4273>(a);
    case 587:
        return RadicalInverseSpecialized<4283>(a);
    case 588:
        return RadicalInverseSpecialized<4289>(a);
    case 589:
        return RadicalInverseSpecialized<4297>(a);
    case 590:
        return RadicalInverseSpecialized<4327>(a);
    case 591:
        return RadicalInverseSpecialized<4337>(a);
    case 592:
        return RadicalInverseSpecialized<4339>(a);
    case 593:
        return RadicalInverseSpecialized<4349>(a);
    case 594:
        return RadicalInverseSpecialized<4357>(a);
    case 595:
        return RadicalInverseSpecialized<4363>(a);
    case 596:
        return RadicalInverseSpecialized<4373>(a);
    case 597:
        return RadicalInverseSpecialized<4391>(a);
    case 598:
        return RadicalInverseSpecialized<4397>(a);
    case 599:
        return RadicalInverseSpecialized<4409>(a);
    case 600:
        return RadicalInverseSpecialized<4421>(a);
    case 601:
        return RadicalInverseSpecialized<4423>(a);
    case 602:
        return RadicalInverseSpecialized<4441>(a);
    case 603:
        return RadicalInverseSpecialized<4447>(a);
    case 604:
        return RadicalInverseSpecialized<4451>(a);
    case 605:
        return RadicalInverseSpecialized<4457>(a);
    case 606:
        return RadicalInverseSpecialized<4463>(a);
    case 607:
        return RadicalInverseSpecialized<4481>(a);
    case 608:
        return RadicalInverseSpecialized<4483>(a);
    case 609:
        return RadicalInverseSpecialized<4493>(a);
    case 610:
        return RadicalInverseSpecialized<4507>(a);
    case 611:
        return RadicalInverseSpecialized<4513>(a);
    case 612:
        return RadicalInverseSpecialized<4517>(a);
    case 613:
        return RadicalInverseSpecialized<4519>(a);
    case 614:
        return RadicalInverseSpecialized<4523>(a);
    case 615:
        return RadicalInverseSpecialized<4547>(a);
    case 616:
        return RadicalInverseSpecialized<4549>(a);
    case 617:
        return RadicalInverseSpecialized<4561>(a);
    case 618:
        return RadicalInverseSpecialized<4567>(a);
    case 619:
        return RadicalInverseSpecialized<4583>(a);
    case 620:
        return RadicalInverseSpecialized<4591>(a);
    case 621:
        return RadicalInverseSpecialized<4597>(a);
    case 622:
        return RadicalInverseSpecialized<4603>(a);
    case 623:
        return RadicalInverseSpecialized<4621>(a);
    case 624:
        return RadicalInverseSpecialized<4637>(a);
    case 625:
        return RadicalInverseSpecialized<4639>(a);
    case 626:
        return RadicalInverseSpecialized<4643>(a);
    case 627:
        return RadicalInverseSpecialized<4649>(a);
    case 628:
        return RadicalInverseSpecialized<4651>(a);
    case 629:
        return RadicalInverseSpecialized<4657>(a);
    case 630:
        return RadicalInverseSpecialized<4663>(a);
    case 631:
        return RadicalInverseSpecialized<4673>(a);
    case 632:
        return RadicalInverseSpecialized<4679>(a);
    case 633:
        return RadicalInverseSpecialized<4691>(a);
    case 634:
        return RadicalInverseSpecialized<4703>(a);
    case 635:
        return RadicalInverseSpecialized<4721>(a);
    case 636:
        return RadicalInverseSpecialized<4723>(a);
    case 637:
        return RadicalInverseSpecialized<4729>(a);
    case 638:
        return RadicalInverseSpecialized<4733>(a);
    case 639:
        return RadicalInverseSpecialized<4751>(a);
    case 640:
        return RadicalInverseSpecialized<4759>(a);
    case 641:
        return RadicalInverseSpecialized<4783>(a);
    case 642:
        return RadicalInverseSpecialized<4787>(a);
    case 643:
        return RadicalInverseSpecialized<4789>(a);
    case 644:
        return RadicalInverseSpecialized<4793>(a);
    case 645:
        return RadicalInverseSpecialized<4799>(a);
    case 646:
        return RadicalInverseSpecialized<4801>(a);
    case 647:
        return RadicalInverseSpecialized<4813>(a);
    case 648:
        return RadicalInverseSpecialized<4817>(a);
    case 649:
        return RadicalInverseSpecialized<4831>(a);
    case 650:
        return RadicalInverseSpecialized<4861>(a);
    case 651:
        return RadicalInverseSpecialized<4871>(a);
    case 652:
        return RadicalInverseSpecialized<4877>(a);
    case 653:
        return RadicalInverseSpecialized<4889>(a);
    case 654:
        return RadicalInverseSpecialized<4903>(a);
    case 655:
        return RadicalInverseSpecialized<4909>(a);
    case 656:
        return RadicalInverseSpecialized<4919>(a);
    case 657:
        return RadicalInverseSpecialized<4931>(a);
    case 658:
        return RadicalInverseSpecialized<4933>(a);
    case 659:
        return RadicalInverseSpecialized<4937>(a);
    case 660:
        return RadicalInverseSpecialized<4943>(a);
    case 661:
        return RadicalInverseSpecialized<4951>(a);
    case 662:
        return RadicalInverseSpecialized<4957>(a);
    case 663:
        return RadicalInverseSpecialized<4967>(a);
    case 664:
        return RadicalInverseSpecialized<4969>(a);
    case 665:
        return RadicalInverseSpecialized<4973>(a);
    case 666:
        return RadicalInverseSpecialized<4987>(a);
    case 667:
        return RadicalInverseSpecialized<4993>(a);
    case 668:
        return RadicalInverseSpecialized<4999>(a);
    case 669:
        return RadicalInverseSpecialized<5003>(a);
    case 670:
        return RadicalInverseSpecialized<5009>(a);
    case 671:
        return RadicalInverseSpecialized<5011>(a);
    case 672:
        return RadicalInverseSpecialized<5021>(a);
    case 673:
        return RadicalInverseSpecialized<5023>(a);
    case 674:
        return RadicalInverseSpecialized<5039>(a);
    case 675:
        return RadicalInverseSpecialized<5051>(a);
    case 676:
        return RadicalInverseSpecialized<5059>(a);
    case 677:
        return RadicalInverseSpecialized<5077>(a);
    case 678:
        return RadicalInverseSpecialized<5081>(a);
    case 679:
        return RadicalInverseSpecialized<5087>(a);
    case 680:
        return RadicalInverseSpecialized<5099>(a);
    case 681:
        return RadicalInverseSpecialized<5101>(a);
    case 682:
        return RadicalInverseSpecialized<5107>(a);
    case 683:
        return RadicalInverseSpecialized<5113>(a);
    case 684:
        return RadicalInverseSpecialized<5119>(a);
    case 685:
        return RadicalInverseSpecialized<5147>(a);
    case 686:
        return RadicalInverseSpecialized<5153>(a);
    case 687:
        return RadicalInverseSpecialized<5167>(a);
    case 688:
        return RadicalInverseSpecialized<5171>(a);
    case 689:
        return RadicalInverseSpecialized<5179>(a);
    case 690:
        return RadicalInverseSpecialized<5189>(a);
    case 691:
        return RadicalInverseSpecialized<5197>(a);
    case 692:
        return RadicalInverseSpecialized<5209>(a);
    case 693:
        return RadicalInverseSpecialized<5227>(a);
    case 694:
        return RadicalInverseSpecialized<5231>(a);
    case 695:
        return RadicalInverseSpecialized<5233>(a);
    case 696:
        return RadicalInverseSpecialized<5237>(a);
    case 697:
        return RadicalInverseSpecialized<5261>(a);
    case 698:
        return RadicalInverseSpecialized<5273>(a);
    case 699:
        return RadicalInverseSpecialized<5279>(a);
    case 700:
        return RadicalInverseSpecialized<5281>(a);
    case 701:
        return RadicalInverseSpecialized<5297>(a);
    case 702:
        return RadicalInverseSpecialized<5303>(a);
    case 703:
        return RadicalInverseSpecialized<5309>(a);
    case 704:
        return RadicalInverseSpecialized<5323>(a);
    case 705:
        return RadicalInverseSpecialized<5333>(a);
    case 706:
        return RadicalInverseSpecialized<5347>(a);
    case 707:
        return RadicalInverseSpecialized<5351>(a);
    case 708:
        return RadicalInverseSpecialized<5381>(a);
    case 709:
        return RadicalInverseSpecialized<5387>(a);
    case 710:
        return RadicalInverseSpecialized<5393>(a);
    case 711:
        return RadicalInverseSpecialized<5399>(a);
    case 712:
        return RadicalInverseSpecialized<5407>(a);
    case 713:
        return RadicalInverseSpecialized<5413>(a);
    case 714:
        return RadicalInverseSpecialized<5417>(a);
    case 715:
        return RadicalInverseSpecialized<5419>(a);
    case 716:
        return RadicalInverseSpecialized<5431>(a);
    case 717:
        return RadicalInverseSpecialized<5437>(a);
    case 718:
        return RadicalInverseSpecialized<5441>(a);
    case 719:
        return RadicalInverseSpecialized<5443>(a);
    case 720:
        return RadicalInverseSpecialized<5449>(a);
    case 721:
        return RadicalInverseSpecialized<5471>(a);
    case 722:
        return RadicalInverseSpecialized<5477>(a);
    case 723:
        return RadicalInverseSpecialized<5479>(a);
    case 724:
        return RadicalInverseSpecialized<5483>(a);
    case 725:
        return RadicalInverseSpecialized<5501>(a);
    case 726:
        return RadicalInverseSpecialized<5503>(a);
    case 727:
        return RadicalInverseSpecialized<5507>(a);
    case 728:
        return RadicalInverseSpecialized<5519>(a);
    case 729:
        return RadicalInverseSpecialized<5521>(a);
    case 730:
        return RadicalInverseSpecialized<5527>(a);
    case 731:
        return RadicalInverseSpecialized<5531>(a);
    case 732:
        return RadicalInverseSpecialized<5557>(a);
    case 733:
        return RadicalInverseSpecialized<5563>(a);
    case 734:
        return RadicalInverseSpecialized<5569>(a);
    case 735:
        return RadicalInverseSpecialized<5573>(a);
    case 736:
        return RadicalInverseSpecialized<5581>(a);
    case 737:
        return RadicalInverseSpecialized<5591>(a);
    case 738:
        return RadicalInverseSpecialized<5623>(a);
    case 739:
        return RadicalInverseSpecialized<5639>(a);
    case 740:
        return RadicalInverseSpecialized<5641>(a);
    case 741:
        return RadicalInverseSpecialized<5647>(a);
    case 742:
        return RadicalInverseSpecialized<5651>(a);
    case 743:
        return RadicalInverseSpecialized<5653>(a);
    case 744:
        return RadicalInverseSpecialized<5657>(a);
    case 745:
        return RadicalInverseSpecialized<5659>(a);
    case 746:
        return RadicalInverseSpecialized<5669>(a);
    case 747:
        return RadicalInverseSpecialized<5683>(a);
    case 748:
        return RadicalInverseSpecialized<5689>(a);
    case 749:
        return RadicalInverseSpecialized<5693>(a);
    case 750:
        return RadicalInverseSpecialized<5701>(a);
    case 751:
        return RadicalInverseSpecialized<5711>(a);
    case 752:
        return RadicalInverseSpecialized<5717>(a);
    case 753:
        return RadicalInverseSpecialized<5737>(a);
    case 754:
        return RadicalInverseSpecialized<5741>(a);
    case 755:
        return RadicalInverseSpecialized<5743>(a);
    case 756:
        return RadicalInverseSpecialized<5749>(a);
    case 757:
        return RadicalInverseSpecialized<5779>(a);
    case 758:
        return RadicalInverseSpecialized<5783>(a);
    case 759:
        return RadicalInverseSpecialized<5791>(a);
    case 760:
        return RadicalInverseSpecialized<5801>(a);
    case 761:
        return RadicalInverseSpecialized<5807>(a);
    case 762:
        return RadicalInverseSpecialized<5813>(a);
    case 763:
        return RadicalInverseSpecialized<5821>(a);
    case 764:
        return RadicalInverseSpecialized<5827>(a);
    case 765:
        return RadicalInverseSpecialized<5839>(a);
    case 766:
        return RadicalInverseSpecialized<5843>(a);
    case 767:
        return RadicalInverseSpecialized<5849>(a);
    case 768:
        return RadicalInverseSpecialized<5851>(a);
    case 769:
        return RadicalInverseSpecialized<5857>(a);
    case 770:
        return RadicalInverseSpecialized<5861>(a);
    case 771:
        return RadicalInverseSpecialized<5867>(a);
    case 772:
        return RadicalInverseSpecialized<5869>(a);
    case 773:
        return RadicalInverseSpecialized<5879>(a);
    case 774:
        return RadicalInverseSpecialized<5881>(a);
    case 775:
        return RadicalInverseSpecialized<5897>(a);
    case 776:
        return RadicalInverseSpecialized<5903>(a);
    case 777:
        return RadicalInverseSpecialized<5923>(a);
    case 778:
        return RadicalInverseSpecialized<5927>(a);
    case 779:
        return RadicalInverseSpecialized<5939>(a);
    case 780:
        return RadicalInverseSpecialized<5953>(a);
    case 781:
        return RadicalInverseSpecialized<5981>(a);
    case 782:
        return RadicalInverseSpecialized<5987>(a);
    case 783:
        return RadicalInverseSpecialized<6007>(a);
    case 784:
        return RadicalInverseSpecialized<6011>(a);
    case 785:
        return RadicalInverseSpecialized<6029>(a);
    case 786:
        return RadicalInverseSpecialized<6037>(a);
    case 787:
        return RadicalInverseSpecialized<6043>(a);
    case 788:
        return RadicalInverseSpecialized<6047>(a);
    case 789:
        return RadicalInverseSpecialized<6053>(a);
    case 790:
        return RadicalInverseSpecialized<6067>(a);
    case 791:
        return RadicalInverseSpecialized<6073>(a);
    case 792:
        return RadicalInverseSpecialized<6079>(a);
    case 793:
        return RadicalInverseSpecialized<6089>(a);
    case 794:
        return RadicalInverseSpecialized<6091>(a);
    case 795:
        return RadicalInverseSpecialized<6101>(a);
    case 796:
        return RadicalInverseSpecialized<6113>(a);
    case 797:
        return RadicalInverseSpecialized<6121>(a);
    case 798:
        return RadicalInverseSpecialized<6131>(a);
    case 799:
        return RadicalInverseSpecialized<6133>(a);
    case 800:
        return RadicalInverseSpecialized<6143>(a);
    case 801:
        return RadicalInverseSpecialized<6151>(a);
    case 802:
        return RadicalInverseSpecialized<6163>(a);
    case 803:
        return RadicalInverseSpecialized<6173>(a);
    case 804:
        return RadicalInverseSpecialized<6197>(a);
    case 805:
        return RadicalInverseSpecialized<6199>(a);
    case 806:
        return RadicalInverseSpecialized<6203>(a);
    case 807:
        return RadicalInverseSpecialized<6211>(a);
    case 808:
        return RadicalInverseSpecialized<6217>(a);
    case 809:
        return RadicalInverseSpecialized<6221>(a);
    case 810:
        return RadicalInverseSpecialized<6229>(a);
    case 811:
        return RadicalInverseSpecialized<6247>(a);
    case 812:
        return RadicalInverseSpecialized<6257>(a);
    case 813:
        return RadicalInverseSpecialized<6263>(a);
    case 814:
        return RadicalInverseSpecialized<6269>(a);
    case 815:
        return RadicalInverseSpecialized<6271>(a);
    case 816:
        return RadicalInverseSpecialized<6277>(a);
    case 817:
        return RadicalInverseSpecialized<6287>(a);
    case 818:
        return RadicalInverseSpecialized<6299>(a);
    case 819:
        return RadicalInverseSpecialized<6301>(a);
    case 820:
        return RadicalInverseSpecialized<6311>(a);
    case 821:
        return RadicalInverseSpecialized<6317>(a);
    case 822:
        return RadicalInverseSpecialized<6323>(a);
    case 823:
        return RadicalInverseSpecialized<6329>(a);
    case 824:
        return RadicalInverseSpecialized<6337>(a);
    case 825:
        return RadicalInverseSpecialized<6343>(a);
    case 826:
        return RadicalInverseSpecialized<6353>(a);
    case 827:
        return RadicalInverseSpecialized<6359>(a);
    case 828:
        return RadicalInverseSpecialized<6361>(a);
    case 829:
        return RadicalInverseSpecialized<6367>(a);
    case 830:
        return RadicalInverseSpecialized<6373>(a);
    case 831:
        return RadicalInverseSpecialized<6379>(a);
    case 832:
        return RadicalInverseSpecialized<6389>(a);
    case 833:
        return RadicalInverseSpecialized<6397>(a);
    case 834:
        return RadicalInverseSpecialized<6421>(a);
    case 835:
        return RadicalInverseSpecialized<6427>(a);
    case 836:
        return RadicalInverseSpecialized<6449>(a);
    case 837:
        return RadicalInverseSpecialized<6451>(a);
    case 838:
        return RadicalInverseSpecialized<6469>(a);
    case 839:
        return RadicalInverseSpecialized<6473>(a);
    case 840:
        return RadicalInverseSpecialized<6481>(a);
    case 841:
        return RadicalInverseSpecialized<6491>(a);
    case 842:
        return RadicalInverseSpecialized<6521>(a);
    case 843:
        return RadicalInverseSpecialized<6529>(a);
    case 844:
        return RadicalInverseSpecialized<6547>(a);
    case 845:
        return RadicalInverseSpecialized<6551>(a);
    case 846:
        return RadicalInverseSpecialized<6553>(a);
    case 847:
        return RadicalInverseSpecialized<6563>(a);
    case 848:
        return RadicalInverseSpecialized<6569>(a);
    case 849:
        return RadicalInverseSpecialized<6571>(a);
    case 850:
        return RadicalInverseSpecialized<6577>(a);
    case 851:
        return RadicalInverseSpecialized<6581>(a);
    case 852:
        return RadicalInverseSpecialized<6599>(a);
    case 853:
        return RadicalInverseSpecialized<6607>(a);
    case 854:
        return RadicalInverseSpecialized<6619>(a);
    case 855:
        return RadicalInverseSpecialized<6637>(a);
    case 856:
        return RadicalInverseSpecialized<6653>(a);
    case 857:
        return RadicalInverseSpecialized<6659>(a);
    case 858:
        return RadicalInverseSpecialized<6661>(a);
    case 859:
        return RadicalInverseSpecialized<6673>(a);
    case 860:
        return RadicalInverseSpecialized<6679>(a);
    case 861:
        return RadicalInverseSpecialized<6689>(a);
    case 862:
        return RadicalInverseSpecialized<6691>(a);
    case 863:
        return RadicalInverseSpecialized<6701>(a);
    case 864:
        return RadicalInverseSpecialized<6703>(a);
    case 865:
        return RadicalInverseSpecialized<6709>(a);
    case 866:
        return RadicalInverseSpecialized<6719>(a);
    case 867:
        return RadicalInverseSpecialized<6733>(a);
    case 868:
        return RadicalInverseSpecialized<6737>(a);
    case 869:
        return RadicalInverseSpecialized<6761>(a);
    case 870:
        return RadicalInverseSpecialized<6763>(a);
    case 871:
        return RadicalInverseSpecialized<6779>(a);
    case 872:
        return RadicalInverseSpecialized<6781>(a);
    case 873:
        return RadicalInverseSpecialized<6791>(a);
    case 874:
        return RadicalInverseSpecialized<6793>(a);
    case 875:
        return RadicalInverseSpecialized<6803>(a);
    case 876:
        return RadicalInverseSpecialized<6823>(a);
    case 877:
        return RadicalInverseSpecialized<6827>(a);
    case 878:
        return RadicalInverseSpecialized<6829>(a);
    case 879:
        return RadicalInverseSpecialized<6833>(a);
    case 880:
        return RadicalInverseSpecialized<6841>(a);
    case 881:
        return RadicalInverseSpecialized<6857>(a);
    case 882:
        return RadicalInverseSpecialized<6863>(a);
    case 883:
        return RadicalInverseSpecialized<6869>(a);
    case 884:
        return RadicalInverseSpecialized<6871>(a);
    case 885:
        return RadicalInverseSpecialized<6883>(a);
    case 886:
        return RadicalInverseSpecialized<6899>(a);
    case 887:
        return RadicalInverseSpecialized<6907>(a);
    case 888:
        return RadicalInverseSpecialized<6911>(a);
    case 889:
        return RadicalInverseSpecialized<6917>(a);
    case 890:
        return RadicalInverseSpecialized<6947>(a);
    case 891:
        return RadicalInverseSpecialized<6949>(a);
    case 892:
        return RadicalInverseSpecialized<6959>(a);
    case 893:
        return RadicalInverseSpecialized<6961>(a);
    case 894:
        return RadicalInverseSpecialized<6967>(a);
    case 895:
        return RadicalInverseSpecialized<6971>(a);
    case 896:
        return RadicalInverseSpecialized<6977>(a);
    case 897:
        return RadicalInverseSpecialized<6983>(a);
    case 898:
        return RadicalInverseSpecialized<6991>(a);
    case 899:
        return RadicalInverseSpecialized<6997>(a);
    case 900:
        return RadicalInverseSpecialized<7001>(a);
    case 901:
        return RadicalInverseSpecialized<7013>(a);
    case 902:
        return RadicalInverseSpecialized<7019>(a);
    case 903:
        return RadicalInverseSpecialized<7027>(a);
    case 904:
        return RadicalInverseSpecialized<7039>(a);
    case 905:
        return RadicalInverseSpecialized<7043>(a);
    case 906:
        return RadicalInverseSpecialized<7057>(a);
    case 907:
        return RadicalInverseSpecialized<7069>(a);
    case 908:
        return RadicalInverseSpecialized<7079>(a);
    case 909:
        return RadicalInverseSpecialized<7103>(a);
    case 910:
        return RadicalInverseSpecialized<7109>(a);
    case 911:
        return RadicalInverseSpecialized<7121>(a);
    case 912:
        return RadicalInverseSpecialized<7127>(a);
    case 913:
        return RadicalInverseSpecialized<7129>(a);
    case 914:
        return RadicalInverseSpecialized<7151>(a);
    case 915:
        return RadicalInverseSpecialized<7159>(a);
    case 916:
        return RadicalInverseSpecialized<7177>(a);
    case 917:
        return RadicalInverseSpecialized<7187>(a);
    case 918:
        return RadicalInverseSpecialized<7193>(a);
    case 919:
        return RadicalInverseSpecialized<7207>(a);
    case 920:
        return RadicalInverseSpecialized<7211>(a);
    case 921:
        return RadicalInverseSpecialized<7213>(a);
    case 922:
        return RadicalInverseSpecialized<7219>(a);
    case 923:
        return RadicalInverseSpecialized<7229>(a);
    case 924:
        return RadicalInverseSpecialized<7237>(a);
    case 925:
        return RadicalInverseSpecialized<7243>(a);
    case 926:
        return RadicalInverseSpecialized<7247>(a);
    case 927:
        return RadicalInverseSpecialized<7253>(a);
    case 928:
        return RadicalInverseSpecialized<7283>(a);
    case 929:
        return RadicalInverseSpecialized<7297>(a);
    case 930:
        return RadicalInverseSpecialized<7307>(a);
    case 931:
        return RadicalInverseSpecialized<7309>(a);
    case 932:
        return RadicalInverseSpecialized<7321>(a);
    case 933:
        return RadicalInverseSpecialized<7331>(a);
    case 934:
        return RadicalInverseSpecialized<7333>(a);
    case 935:
        return RadicalInverseSpecialized<7349>(a);
    case 936:
        return RadicalInverseSpecialized<7351>(a);
    case 937:
        return RadicalInverseSpecialized<7369>(a);
    case 938:
        return RadicalInverseSpecialized<7393>(a);
    case 939:
        return RadicalInverseSpecialized<7411>(a);
    case 940:
        return RadicalInverseSpecialized<7417>(a);
    case 941:
        return RadicalInverseSpecialized<7433>(a);
    case 942:
        return RadicalInverseSpecialized<7451>(a);
    case 943:
        return RadicalInverseSpecialized<7457>(a);
    case 944:
        return RadicalInverseSpecialized<7459>(a);
    case 945:
        return RadicalInverseSpecialized<7477>(a);
    case 946:
        return RadicalInverseSpecialized<7481>(a);
    case 947:
        return RadicalInverseSpecialized<7487>(a);
    case 948:
        return RadicalInverseSpecialized<7489>(a);
    case 949:
        return RadicalInverseSpecialized<7499>(a);
    case 950:
        return RadicalInverseSpecialized<7507>(a);
    case 951:
        return RadicalInverseSpecialized<7517>(a);
    case 952:
        return RadicalInverseSpecialized<7523>(a);
    case 953:
        return RadicalInverseSpecialized<7529>(a);
    case 954:
        return RadicalInverseSpecialized<7537>(a);
    case 955:
        return RadicalInverseSpecialized<7541>(a);
    case 956:
        return RadicalInverseSpecialized<7547>(a);
    case 957:
        return RadicalInverseSpecialized<7549>(a);
    case 958:
        return RadicalInverseSpecialized<7559>(a);
    case 959:
        return RadicalInverseSpecialized<7561>(a);
    case 960:
        return RadicalInverseSpecialized<7573>(a);
    case 961:
        return RadicalInverseSpecialized<7577>(a);
    case 962:
        return RadicalInverseSpecialized<7583>(a);
    case 963:
        return RadicalInverseSpecialized<7589>(a);
    case 964:
        return RadicalInverseSpecialized<7591>(a);
    case 965:
        return RadicalInverseSpecialized<7603>(a);
    case 966:
        return RadicalInverseSpecialized<7607>(a);
    case 967:
        return RadicalInverseSpecialized<7621>(a);
    case 968:
        return RadicalInverseSpecialized<7639>(a);
    case 969:
        return RadicalInverseSpecialized<7643>(a);
    case 970:
        return RadicalInverseSpecialized<7649>(a);
    case 971:
        return RadicalInverseSpecialized<7669>(a);
    case 972:
        return RadicalInverseSpecialized<7673>(a);
    case 973:
        return RadicalInverseSpecialized<7681>(a);
    case 974:
        return RadicalInverseSpecialized<7687>(a);
    case 975:
        return RadicalInverseSpecialized<7691>(a);
    case 976:
        return RadicalInverseSpecialized<7699>(a);
    case 977:
        return RadicalInverseSpecialized<7703>(a);
    case 978:
        return RadicalInverseSpecialized<7717>(a);
    case 979:
        return RadicalInverseSpecialized<7723>(a);
    case 980:
        return RadicalInverseSpecialized<7727>(a);
    case 981:
        return RadicalInverseSpecialized<7741>(a);
    case 982:
        return RadicalInverseSpecialized<7753>(a);
    case 983:
        return RadicalInverseSpecialized<7757>(a);
    case 984:
        return RadicalInverseSpecialized<7759>(a);
    case 985:
        return RadicalInverseSpecialized<7789>(a);
    case 986:
        return RadicalInverseSpecialized<7793>(a);
    case 987:
        return RadicalInverseSpecialized<7817>(a);
    case 988:
        return RadicalInverseSpecialized<7823>(a);
    case 989:
        return RadicalInverseSpecialized<7829>(a);
    case 990:
        return RadicalInverseSpecialized<7841>(a);
    case 991:
        return RadicalInverseSpecialized<7853>(a);
    case 992:
        return RadicalInverseSpecialized<7867>(a);
    case 993:
        return RadicalInverseSpecialized<7873>(a);
    case 994:
        return RadicalInverseSpecialized<7877>(a);
    case 995:
        return RadicalInverseSpecialized<7879>(a);
    case 996:
        return RadicalInverseSpecialized<7883>(a);
    case 997:
        return RadicalInverseSpecialized<7901>(a);
    case 998:
        return RadicalInverseSpecialized<7907>(a);
    case 999:
        return RadicalInverseSpecialized<7919>(a);
    case 1000:
        return RadicalInverseSpecialized<7927>(a);
    case 1001:
        return RadicalInverseSpecialized<7933>(a);
    case 1002:
        return RadicalInverseSpecialized<7937>(a);
    case 1003:
        return RadicalInverseSpecialized<7949>(a);
    case 1004:
        return RadicalInverseSpecialized<7951>(a);
    case 1005:
        return RadicalInverseSpecialized<7963>(a);
    case 1006:
        return RadicalInverseSpecialized<7993>(a);
    case 1007:
        return RadicalInverseSpecialized<8009>(a);
    case 1008:
        return RadicalInverseSpecialized<8011>(a);
    case 1009:
        return RadicalInverseSpecialized<8017>(a);
    case 1010:
        return RadicalInverseSpecialized<8039>(a);
    case 1011:
        return RadicalInverseSpecialized<8053>(a);
    case 1012:
        return RadicalInverseSpecialized<8059>(a);
    case 1013:
        return RadicalInverseSpecialized<8069>(a);
    case 1014:
        return RadicalInverseSpecialized<8081>(a);
    case 1015:
        return RadicalInverseSpecialized<8087>(a);
    case 1016:
        return RadicalInverseSpecialized<8089>(a);
    case 1017:
        return RadicalInverseSpecialized<8093>(a);
    case 1018:
        return RadicalInverseSpecialized<8101>(a);
    case 1019:
        return RadicalInverseSpecialized<8111>(a);
    case 1020:
        return RadicalInverseSpecialized<8117>(a);
    case 1021:
        return RadicalInverseSpecialized<8123>(a);
    case 1022:
        return RadicalInverseSpecialized<8147>(a);
    case 1023:
        return RadicalInverseSpecialized<8161>(a);
    default:
        LOG_FATAL("Base %d is >= 1024, the limit of RadicalInverse", baseIndex);
        return 0;
    }
}

pstd::vector<DigitPermutation> *ComputeRadicalInversePermutations(uint32_t seed,
                                                                  Allocator alloc) {
    pstd::vector<DigitPermutation> *perms =
        alloc.new_object<pstd::vector<DigitPermutation>>(alloc);
    perms->resize(PrimeTableSize);
    ParallelFor(0, PrimeTableSize, [&perms, &alloc, seed](int64_t i) {
        (*perms)[i] = DigitPermutation(Primes[i], seed, alloc);
    });
    return perms;
}

Float ScrambledRadicalInverse(int baseIndex, uint64_t a, const DigitPermutation &perm) {
    switch (baseIndex) {
    case 0:
        return ScrambledRadicalInverseSpecialized<2>(a, perm);
    case 1:
        return ScrambledRadicalInverseSpecialized<3>(a, perm);
    case 2:
        return ScrambledRadicalInverseSpecialized<5>(a, perm);
    case 3:
        return ScrambledRadicalInverseSpecialized<7>(a, perm);
    // Remainder of cases for _ScrambledRadicalInverse()_
    case 4:
        return ScrambledRadicalInverseSpecialized<11>(a, perm);
    case 5:
        return ScrambledRadicalInverseSpecialized<13>(a, perm);
    case 6:
        return ScrambledRadicalInverseSpecialized<17>(a, perm);
    case 7:
        return ScrambledRadicalInverseSpecialized<19>(a, perm);
    case 8:
        return ScrambledRadicalInverseSpecialized<23>(a, perm);
    case 9:
        return ScrambledRadicalInverseSpecialized<29>(a, perm);
    case 10:
        return ScrambledRadicalInverseSpecialized<31>(a, perm);
    case 11:
        return ScrambledRadicalInverseSpecialized<37>(a, perm);
    case 12:
        return ScrambledRadicalInverseSpecialized<41>(a, perm);
    case 13:
        return ScrambledRadicalInverseSpecialized<43>(a, perm);
    case 14:
        return ScrambledRadicalInverseSpecialized<47>(a, perm);
    case 15:
        return ScrambledRadicalInverseSpecialized<53>(a, perm);
    case 16:
        return ScrambledRadicalInverseSpecialized<59>(a, perm);
    case 17:
        return ScrambledRadicalInverseSpecialized<61>(a, perm);
    case 18:
        return ScrambledRadicalInverseSpecialized<67>(a, perm);
    case 19:
        return ScrambledRadicalInverseSpecialized<71>(a, perm);
    case 20:
        return ScrambledRadicalInverseSpecialized<73>(a, perm);
    case 21:
        return ScrambledRadicalInverseSpecialized<79>(a, perm);
    case 22:
        return ScrambledRadicalInverseSpecialized<83>(a, perm);
    case 23:
        return ScrambledRadicalInverseSpecialized<89>(a, perm);
    case 24:
        return ScrambledRadicalInverseSpecialized<97>(a, perm);
    case 25:
        return ScrambledRadicalInverseSpecialized<101>(a, perm);
    case 26:
        return ScrambledRadicalInverseSpecialized<103>(a, perm);
    case 27:
        return ScrambledRadicalInverseSpecialized<107>(a, perm);
    case 28:
        return ScrambledRadicalInverseSpecialized<109>(a, perm);
    case 29:
        return ScrambledRadicalInverseSpecialized<113>(a, perm);
    case 30:
        return ScrambledRadicalInverseSpecialized<127>(a, perm);
    case 31:
        return ScrambledRadicalInverseSpecialized<131>(a, perm);
    case 32:
        return ScrambledRadicalInverseSpecialized<137>(a, perm);
    case 33:
        return ScrambledRadicalInverseSpecialized<139>(a, perm);
    case 34:
        return ScrambledRadicalInverseSpecialized<149>(a, perm);
    case 35:
        return ScrambledRadicalInverseSpecialized<151>(a, perm);
    case 36:
        return ScrambledRadicalInverseSpecialized<157>(a, perm);
    case 37:
        return ScrambledRadicalInverseSpecialized<163>(a, perm);
    case 38:
        return ScrambledRadicalInverseSpecialized<167>(a, perm);
    case 39:
        return ScrambledRadicalInverseSpecialized<173>(a, perm);
    case 40:
        return ScrambledRadicalInverseSpecialized<179>(a, perm);
    case 41:
        return ScrambledRadicalInverseSpecialized<181>(a, perm);
    case 42:
        return ScrambledRadicalInverseSpecialized<191>(a, perm);
    case 43:
        return ScrambledRadicalInverseSpecialized<193>(a, perm);
    case 44:
        return ScrambledRadicalInverseSpecialized<197>(a, perm);
    case 45:
        return ScrambledRadicalInverseSpecialized<199>(a, perm);
    case 46:
        return ScrambledRadicalInverseSpecialized<211>(a, perm);
    case 47:
        return ScrambledRadicalInverseSpecialized<223>(a, perm);
    case 48:
        return ScrambledRadicalInverseSpecialized<227>(a, perm);
    case 49:
        return ScrambledRadicalInverseSpecialized<229>(a, perm);
    case 50:
        return ScrambledRadicalInverseSpecialized<233>(a, perm);
    case 51:
        return ScrambledRadicalInverseSpecialized<239>(a, perm);
    case 52:
        return ScrambledRadicalInverseSpecialized<241>(a, perm);
    case 53:
        return ScrambledRadicalInverseSpecialized<251>(a, perm);
    case 54:
        return ScrambledRadicalInverseSpecialized<257>(a, perm);
    case 55:
        return ScrambledRadicalInverseSpecialized<263>(a, perm);
    case 56:
        return ScrambledRadicalInverseSpecialized<269>(a, perm);
    case 57:
        return ScrambledRadicalInverseSpecialized<271>(a, perm);
    case 58:
        return ScrambledRadicalInverseSpecialized<277>(a, perm);
    case 59:
        return ScrambledRadicalInverseSpecialized<281>(a, perm);
    case 60:
        return ScrambledRadicalInverseSpecialized<283>(a, perm);
    case 61:
        return ScrambledRadicalInverseSpecialized<293>(a, perm);
    case 62:
        return ScrambledRadicalInverseSpecialized<307>(a, perm);
    case 63:
        return ScrambledRadicalInverseSpecialized<311>(a, perm);
    case 64:
        return ScrambledRadicalInverseSpecialized<313>(a, perm);
    case 65:
        return ScrambledRadicalInverseSpecialized<317>(a, perm);
    case 66:
        return ScrambledRadicalInverseSpecialized<331>(a, perm);
    case 67:
        return ScrambledRadicalInverseSpecialized<337>(a, perm);
    case 68:
        return ScrambledRadicalInverseSpecialized<347>(a, perm);
    case 69:
        return ScrambledRadicalInverseSpecialized<349>(a, perm);
    case 70:
        return ScrambledRadicalInverseSpecialized<353>(a, perm);
    case 71:
        return ScrambledRadicalInverseSpecialized<359>(a, perm);
    case 72:
        return ScrambledRadicalInverseSpecialized<367>(a, perm);
    case 73:
        return ScrambledRadicalInverseSpecialized<373>(a, perm);
    case 74:
        return ScrambledRadicalInverseSpecialized<379>(a, perm);
    case 75:
        return ScrambledRadicalInverseSpecialized<383>(a, perm);
    case 76:
        return ScrambledRadicalInverseSpecialized<389>(a, perm);
    case 77:
        return ScrambledRadicalInverseSpecialized<397>(a, perm);
    case 78:
        return ScrambledRadicalInverseSpecialized<401>(a, perm);
    case 79:
        return ScrambledRadicalInverseSpecialized<409>(a, perm);
    case 80:
        return ScrambledRadicalInverseSpecialized<419>(a, perm);
    case 81:
        return ScrambledRadicalInverseSpecialized<421>(a, perm);
    case 82:
        return ScrambledRadicalInverseSpecialized<431>(a, perm);
    case 83:
        return ScrambledRadicalInverseSpecialized<433>(a, perm);
    case 84:
        return ScrambledRadicalInverseSpecialized<439>(a, perm);
    case 85:
        return ScrambledRadicalInverseSpecialized<443>(a, perm);
    case 86:
        return ScrambledRadicalInverseSpecialized<449>(a, perm);
    case 87:
        return ScrambledRadicalInverseSpecialized<457>(a, perm);
    case 88:
        return ScrambledRadicalInverseSpecialized<461>(a, perm);
    case 89:
        return ScrambledRadicalInverseSpecialized<463>(a, perm);
    case 90:
        return ScrambledRadicalInverseSpecialized<467>(a, perm);
    case 91:
        return ScrambledRadicalInverseSpecialized<479>(a, perm);
    case 92:
        return ScrambledRadicalInverseSpecialized<487>(a, perm);
    case 93:
        return ScrambledRadicalInverseSpecialized<491>(a, perm);
    case 94:
        return ScrambledRadicalInverseSpecialized<499>(a, perm);
    case 95:
        return ScrambledRadicalInverseSpecialized<503>(a, perm);
    case 96:
        return ScrambledRadicalInverseSpecialized<509>(a, perm);
    case 97:
        return ScrambledRadicalInverseSpecialized<521>(a, perm);
    case 98:
        return ScrambledRadicalInverseSpecialized<523>(a, perm);
    case 99:
        return ScrambledRadicalInverseSpecialized<541>(a, perm);
    case 100:
        return ScrambledRadicalInverseSpecialized<547>(a, perm);
    case 101:
        return ScrambledRadicalInverseSpecialized<557>(a, perm);
    case 102:
        return ScrambledRadicalInverseSpecialized<563>(a, perm);
    case 103:
        return ScrambledRadicalInverseSpecialized<569>(a, perm);
    case 104:
        return ScrambledRadicalInverseSpecialized<571>(a, perm);
    case 105:
        return ScrambledRadicalInverseSpecialized<577>(a, perm);
    case 106:
        return ScrambledRadicalInverseSpecialized<587>(a, perm);
    case 107:
        return ScrambledRadicalInverseSpecialized<593>(a, perm);
    case 108:
        return ScrambledRadicalInverseSpecialized<599>(a, perm);
    case 109:
        return ScrambledRadicalInverseSpecialized<601>(a, perm);
    case 110:
        return ScrambledRadicalInverseSpecialized<607>(a, perm);
    case 111:
        return ScrambledRadicalInverseSpecialized<613>(a, perm);
    case 112:
        return ScrambledRadicalInverseSpecialized<617>(a, perm);
    case 113:
        return ScrambledRadicalInverseSpecialized<619>(a, perm);
    case 114:
        return ScrambledRadicalInverseSpecialized<631>(a, perm);
    case 115:
        return ScrambledRadicalInverseSpecialized<641>(a, perm);
    case 116:
        return ScrambledRadicalInverseSpecialized<643>(a, perm);
    case 117:
        return ScrambledRadicalInverseSpecialized<647>(a, perm);
    case 118:
        return ScrambledRadicalInverseSpecialized<653>(a, perm);
    case 119:
        return ScrambledRadicalInverseSpecialized<659>(a, perm);
    case 120:
        return ScrambledRadicalInverseSpecialized<661>(a, perm);
    case 121:
        return ScrambledRadicalInverseSpecialized<673>(a, perm);
    case 122:
        return ScrambledRadicalInverseSpecialized<677>(a, perm);
    case 123:
        return ScrambledRadicalInverseSpecialized<683>(a, perm);
    case 124:
        return ScrambledRadicalInverseSpecialized<691>(a, perm);
    case 125:
        return ScrambledRadicalInverseSpecialized<701>(a, perm);
    case 126:
        return ScrambledRadicalInverseSpecialized<709>(a, perm);
    case 127:
        return ScrambledRadicalInverseSpecialized<719>(a, perm);
    case 128:
        return ScrambledRadicalInverseSpecialized<727>(a, perm);
    case 129:
        return ScrambledRadicalInverseSpecialized<733>(a, perm);
    case 130:
        return ScrambledRadicalInverseSpecialized<739>(a, perm);
    case 131:
        return ScrambledRadicalInverseSpecialized<743>(a, perm);
    case 132:
        return ScrambledRadicalInverseSpecialized<751>(a, perm);
    case 133:
        return ScrambledRadicalInverseSpecialized<757>(a, perm);
    case 134:
        return ScrambledRadicalInverseSpecialized<761>(a, perm);
    case 135:
        return ScrambledRadicalInverseSpecialized<769>(a, perm);
    case 136:
        return ScrambledRadicalInverseSpecialized<773>(a, perm);
    case 137:
        return ScrambledRadicalInverseSpecialized<787>(a, perm);
    case 138:
        return ScrambledRadicalInverseSpecialized<797>(a, perm);
    case 139:
        return ScrambledRadicalInverseSpecialized<809>(a, perm);
    case 140:
        return ScrambledRadicalInverseSpecialized<811>(a, perm);
    case 141:
        return ScrambledRadicalInverseSpecialized<821>(a, perm);
    case 142:
        return ScrambledRadicalInverseSpecialized<823>(a, perm);
    case 143:
        return ScrambledRadicalInverseSpecialized<827>(a, perm);
    case 144:
        return ScrambledRadicalInverseSpecialized<829>(a, perm);
    case 145:
        return ScrambledRadicalInverseSpecialized<839>(a, perm);
    case 146:
        return ScrambledRadicalInverseSpecialized<853>(a, perm);
    case 147:
        return ScrambledRadicalInverseSpecialized<857>(a, perm);
    case 148:
        return ScrambledRadicalInverseSpecialized<859>(a, perm);
    case 149:
        return ScrambledRadicalInverseSpecialized<863>(a, perm);
    case 150:
        return ScrambledRadicalInverseSpecialized<877>(a, perm);
    case 151:
        return ScrambledRadicalInverseSpecialized<881>(a, perm);
    case 152:
        return ScrambledRadicalInverseSpecialized<883>(a, perm);
    case 153:
        return ScrambledRadicalInverseSpecialized<887>(a, perm);
    case 154:
        return ScrambledRadicalInverseSpecialized<907>(a, perm);
    case 155:
        return ScrambledRadicalInverseSpecialized<911>(a, perm);
    case 156:
        return ScrambledRadicalInverseSpecialized<919>(a, perm);
    case 157:
        return ScrambledRadicalInverseSpecialized<929>(a, perm);
    case 158:
        return ScrambledRadicalInverseSpecialized<937>(a, perm);
    case 159:
        return ScrambledRadicalInverseSpecialized<941>(a, perm);
    case 160:
        return ScrambledRadicalInverseSpecialized<947>(a, perm);
    case 161:
        return ScrambledRadicalInverseSpecialized<953>(a, perm);
    case 162:
        return ScrambledRadicalInverseSpecialized<967>(a, perm);
    case 163:
        return ScrambledRadicalInverseSpecialized<971>(a, perm);
    case 164:
        return ScrambledRadicalInverseSpecialized<977>(a, perm);
    case 165:
        return ScrambledRadicalInverseSpecialized<983>(a, perm);
    case 166:
        return ScrambledRadicalInverseSpecialized<991>(a, perm);
    case 167:
        return ScrambledRadicalInverseSpecialized<997>(a, perm);
    case 168:
        return ScrambledRadicalInverseSpecialized<1009>(a, perm);
    case 169:
        return ScrambledRadicalInverseSpecialized<1013>(a, perm);
    case 170:
        return ScrambledRadicalInverseSpecialized<1019>(a, perm);
    case 171:
        return ScrambledRadicalInverseSpecialized<1021>(a, perm);
    case 172:
        return ScrambledRadicalInverseSpecialized<1031>(a, perm);
    case 173:
        return ScrambledRadicalInverseSpecialized<1033>(a, perm);
    case 174:
        return ScrambledRadicalInverseSpecialized<1039>(a, perm);
    case 175:
        return ScrambledRadicalInverseSpecialized<1049>(a, perm);
    case 176:
        return ScrambledRadicalInverseSpecialized<1051>(a, perm);
    case 177:
        return ScrambledRadicalInverseSpecialized<1061>(a, perm);
    case 178:
        return ScrambledRadicalInverseSpecialized<1063>(a, perm);
    case 179:
        return ScrambledRadicalInverseSpecialized<1069>(a, perm);
    case 180:
        return ScrambledRadicalInverseSpecialized<1087>(a, perm);
    case 181:
        return ScrambledRadicalInverseSpecialized<1091>(a, perm);
    case 182:
        return ScrambledRadicalInverseSpecialized<1093>(a, perm);
    case 183:
        return ScrambledRadicalInverseSpecialized<1097>(a, perm);
    case 184:
        return ScrambledRadicalInverseSpecialized<1103>(a, perm);
    case 185:
        return ScrambledRadicalInverseSpecialized<1109>(a, perm);
    case 186:
        return ScrambledRadicalInverseSpecialized<1117>(a, perm);
    case 187:
        return ScrambledRadicalInverseSpecialized<1123>(a, perm);
    case 188:
        return ScrambledRadicalInverseSpecialized<1129>(a, perm);
    case 189:
        return ScrambledRadicalInverseSpecialized<1151>(a, perm);
    case 190:
        return ScrambledRadicalInverseSpecialized<1153>(a, perm);
    case 191:
        return ScrambledRadicalInverseSpecialized<1163>(a, perm);
    case 192:
        return ScrambledRadicalInverseSpecialized<1171>(a, perm);
    case 193:
        return ScrambledRadicalInverseSpecialized<1181>(a, perm);
    case 194:
        return ScrambledRadicalInverseSpecialized<1187>(a, perm);
    case 195:
        return ScrambledRadicalInverseSpecialized<1193>(a, perm);
    case 196:
        return ScrambledRadicalInverseSpecialized<1201>(a, perm);
    case 197:
        return ScrambledRadicalInverseSpecialized<1213>(a, perm);
    case 198:
        return ScrambledRadicalInverseSpecialized<1217>(a, perm);
    case 199:
        return ScrambledRadicalInverseSpecialized<1223>(a, perm);
    case 200:
        return ScrambledRadicalInverseSpecialized<1229>(a, perm);
    case 201:
        return ScrambledRadicalInverseSpecialized<1231>(a, perm);
    case 202:
        return ScrambledRadicalInverseSpecialized<1237>(a, perm);
    case 203:
        return ScrambledRadicalInverseSpecialized<1249>(a, perm);
    case 204:
        return ScrambledRadicalInverseSpecialized<1259>(a, perm);
    case 205:
        return ScrambledRadicalInverseSpecialized<1277>(a, perm);
    case 206:
        return ScrambledRadicalInverseSpecialized<1279>(a, perm);
    case 207:
        return ScrambledRadicalInverseSpecialized<1283>(a, perm);
    case 208:
        return ScrambledRadicalInverseSpecialized<1289>(a, perm);
    case 209:
        return ScrambledRadicalInverseSpecialized<1291>(a, perm);
    case 210:
        return ScrambledRadicalInverseSpecialized<1297>(a, perm);
    case 211:
        return ScrambledRadicalInverseSpecialized<1301>(a, perm);
    case 212:
        return ScrambledRadicalInverseSpecialized<1303>(a, perm);
    case 213:
        return ScrambledRadicalInverseSpecialized<1307>(a, perm);
    case 214:
        return ScrambledRadicalInverseSpecialized<1319>(a, perm);
    case 215:
        return ScrambledRadicalInverseSpecialized<1321>(a, perm);
    case 216:
        return ScrambledRadicalInverseSpecialized<1327>(a, perm);
    case 217:
        return ScrambledRadicalInverseSpecialized<1361>(a, perm);
    case 218:
        return ScrambledRadicalInverseSpecialized<1367>(a, perm);
    case 219:
        return ScrambledRadicalInverseSpecialized<1373>(a, perm);
    case 220:
        return ScrambledRadicalInverseSpecialized<1381>(a, perm);
    case 221:
        return ScrambledRadicalInverseSpecialized<1399>(a, perm);
    case 222:
        return ScrambledRadicalInverseSpecialized<1409>(a, perm);
    case 223:
        return ScrambledRadicalInverseSpecialized<1423>(a, perm);
    case 224:
        return ScrambledRadicalInverseSpecialized<1427>(a, perm);
    case 225:
        return ScrambledRadicalInverseSpecialized<1429>(a, perm);
    case 226:
        return ScrambledRadicalInverseSpecialized<1433>(a, perm);
    case 227:
        return ScrambledRadicalInverseSpecialized<1439>(a, perm);
    case 228:
        return ScrambledRadicalInverseSpecialized<1447>(a, perm);
    case 229:
        return ScrambledRadicalInverseSpecialized<1451>(a, perm);
    case 230:
        return ScrambledRadicalInverseSpecialized<1453>(a, perm);
    case 231:
        return ScrambledRadicalInverseSpecialized<1459>(a, perm);
    case 232:
        return ScrambledRadicalInverseSpecialized<1471>(a, perm);
    case 233:
        return ScrambledRadicalInverseSpecialized<1481>(a, perm);
    case 234:
        return ScrambledRadicalInverseSpecialized<1483>(a, perm);
    case 235:
        return ScrambledRadicalInverseSpecialized<1487>(a, perm);
    case 236:
        return ScrambledRadicalInverseSpecialized<1489>(a, perm);
    case 237:
        return ScrambledRadicalInverseSpecialized<1493>(a, perm);
    case 238:
        return ScrambledRadicalInverseSpecialized<1499>(a, perm);
    case 239:
        return ScrambledRadicalInverseSpecialized<1511>(a, perm);
    case 240:
        return ScrambledRadicalInverseSpecialized<1523>(a, perm);
    case 241:
        return ScrambledRadicalInverseSpecialized<1531>(a, perm);
    case 242:
        return ScrambledRadicalInverseSpecialized<1543>(a, perm);
    case 243:
        return ScrambledRadicalInverseSpecialized<1549>(a, perm);
    case 244:
        return ScrambledRadicalInverseSpecialized<1553>(a, perm);
    case 245:
        return ScrambledRadicalInverseSpecialized<1559>(a, perm);
    case 246:
        return ScrambledRadicalInverseSpecialized<1567>(a, perm);
    case 247:
        return ScrambledRadicalInverseSpecialized<1571>(a, perm);
    case 248:
        return ScrambledRadicalInverseSpecialized<1579>(a, perm);
    case 249:
        return ScrambledRadicalInverseSpecialized<1583>(a, perm);
    case 250:
        return ScrambledRadicalInverseSpecialized<1597>(a, perm);
    case 251:
        return ScrambledRadicalInverseSpecialized<1601>(a, perm);
    case 252:
        return ScrambledRadicalInverseSpecialized<1607>(a, perm);
    case 253:
        return ScrambledRadicalInverseSpecialized<1609>(a, perm);
    case 254:
        return ScrambledRadicalInverseSpecialized<1613>(a, perm);
    case 255:
        return ScrambledRadicalInverseSpecialized<1619>(a, perm);
    case 256:
        return ScrambledRadicalInverseSpecialized<1621>(a, perm);
    case 257:
        return ScrambledRadicalInverseSpecialized<1627>(a, perm);
    case 258:
        return ScrambledRadicalInverseSpecialized<1637>(a, perm);
    case 259:
        return ScrambledRadicalInverseSpecialized<1657>(a, perm);
    case 260:
        return ScrambledRadicalInverseSpecialized<1663>(a, perm);
    case 261:
        return ScrambledRadicalInverseSpecialized<1667>(a, perm);
    case 262:
        return ScrambledRadicalInverseSpecialized<1669>(a, perm);
    case 263:
        return ScrambledRadicalInverseSpecialized<1693>(a, perm);
    case 264:
        return ScrambledRadicalInverseSpecialized<1697>(a, perm);
    case 265:
        return ScrambledRadicalInverseSpecialized<1699>(a, perm);
    case 266:
        return ScrambledRadicalInverseSpecialized<1709>(a, perm);
    case 267:
        return ScrambledRadicalInverseSpecialized<1721>(a, perm);
    case 268:
        return ScrambledRadicalInverseSpecialized<1723>(a, perm);
    case 269:
        return ScrambledRadicalInverseSpecialized<1733>(a, perm);
    case 270:
        return ScrambledRadicalInverseSpecialized<1741>(a, perm);
    case 271:
        return ScrambledRadicalInverseSpecialized<1747>(a, perm);
    case 272:
        return ScrambledRadicalInverseSpecialized<1753>(a, perm);
    case 273:
        return ScrambledRadicalInverseSpecialized<1759>(a, perm);
    case 274:
        return ScrambledRadicalInverseSpecialized<1777>(a, perm);
    case 275:
        return ScrambledRadicalInverseSpecialized<1783>(a, perm);
    case 276:
        return ScrambledRadicalInverseSpecialized<1787>(a, perm);
    case 277:
        return ScrambledRadicalInverseSpecialized<1789>(a, perm);
    case 278:
        return ScrambledRadicalInverseSpecialized<1801>(a, perm);
    case 279:
        return ScrambledRadicalInverseSpecialized<1811>(a, perm);
    case 280:
        return ScrambledRadicalInverseSpecialized<1823>(a, perm);
    case 281:
        return ScrambledRadicalInverseSpecialized<1831>(a, perm);
    case 282:
        return ScrambledRadicalInverseSpecialized<1847>(a, perm);
    case 283:
        return ScrambledRadicalInverseSpecialized<1861>(a, perm);
    case 284:
        return ScrambledRadicalInverseSpecialized<1867>(a, perm);
    case 285:
        return ScrambledRadicalInverseSpecialized<1871>(a, perm);
    case 286:
        return ScrambledRadicalInverseSpecialized<1873>(a, perm);
    case 287:
        return ScrambledRadicalInverseSpecialized<1877>(a, perm);
    case 288:
        return ScrambledRadicalInverseSpecialized<1879>(a, perm);
    case 289:
        return ScrambledRadicalInverseSpecialized<1889>(a, perm);
    case 290:
        return ScrambledRadicalInverseSpecialized<1901>(a, perm);
    case 291:
        return ScrambledRadicalInverseSpecialized<1907>(a, perm);
    case 292:
        return ScrambledRadicalInverseSpecialized<1913>(a, perm);
    case 293:
        return ScrambledRadicalInverseSpecialized<1931>(a, perm);
    case 294:
        return ScrambledRadicalInverseSpecialized<1933>(a, perm);
    case 295:
        return ScrambledRadicalInverseSpecialized<1949>(a, perm);
    case 296:
        return ScrambledRadicalInverseSpecialized<1951>(a, perm);
    case 297:
        return ScrambledRadicalInverseSpecialized<1973>(a, perm);
    case 298:
        return ScrambledRadicalInverseSpecialized<1979>(a, perm);
    case 299:
        return ScrambledRadicalInverseSpecialized<1987>(a, perm);
    case 300:
        return ScrambledRadicalInverseSpecialized<1993>(a, perm);
    case 301:
        return ScrambledRadicalInverseSpecialized<1997>(a, perm);
    case 302:
        return ScrambledRadicalInverseSpecialized<1999>(a, perm);
    case 303:
        return ScrambledRadicalInverseSpecialized<2003>(a, perm);
    case 304:
        return ScrambledRadicalInverseSpecialized<2011>(a, perm);
    case 305:
        return ScrambledRadicalInverseSpecialized<2017>(a, perm);
    case 306:
        return ScrambledRadicalInverseSpecialized<2027>(a, perm);
    case 307:
        return ScrambledRadicalInverseSpecialized<2029>(a, perm);
    case 308:
        return ScrambledRadicalInverseSpecialized<2039>(a, perm);
    case 309:
        return ScrambledRadicalInverseSpecialized<2053>(a, perm);
    case 310:
        return ScrambledRadicalInverseSpecialized<2063>(a, perm);
    case 311:
        return ScrambledRadicalInverseSpecialized<2069>(a, perm);
    case 312:
        return ScrambledRadicalInverseSpecialized<2081>(a, perm);
    case 313:
        return ScrambledRadicalInverseSpecialized<2083>(a, perm);
    case 314:
        return ScrambledRadicalInverseSpecialized<2087>(a, perm);
    case 315:
        return ScrambledRadicalInverseSpecialized<2089>(a, perm);
    case 316:
        return ScrambledRadicalInverseSpecialized<2099>(a, perm);
    case 317:
        return ScrambledRadicalInverseSpecialized<2111>(a, perm);
    case 318:
        return ScrambledRadicalInverseSpecialized<2113>(a, perm);
    case 319:
        return ScrambledRadicalInverseSpecialized<2129>(a, perm);
    case 320:
        return ScrambledRadicalInverseSpecialized<2131>(a, perm);
    case 321:
        return ScrambledRadicalInverseSpecialized<2137>(a, perm);
    case 322:
        return ScrambledRadicalInverseSpecialized<2141>(a, perm);
    case 323:
        return ScrambledRadicalInverseSpecialized<2143>(a, perm);
    case 324:
        return ScrambledRadicalInverseSpecialized<2153>(a, perm);
    case 325:
        return ScrambledRadicalInverseSpecialized<2161>(a, perm);
    case 326:
        return ScrambledRadicalInverseSpecialized<2179>(a, perm);
    case 327:
        return ScrambledRadicalInverseSpecialized<2203>(a, perm);
    case 328:
        return ScrambledRadicalInverseSpecialized<2207>(a, perm);
    case 329:
        return ScrambledRadicalInverseSpecialized<2213>(a, perm);
    case 330:
        return ScrambledRadicalInverseSpecialized<2221>(a, perm);
    case 331:
        return ScrambledRadicalInverseSpecialized<2237>(a, perm);
    case 332:
        return ScrambledRadicalInverseSpecialized<2239>(a, perm);
    case 333:
        return ScrambledRadicalInverseSpecialized<2243>(a, perm);
    case 334:
        return ScrambledRadicalInverseSpecialized<2251>(a, perm);
    case 335:
        return ScrambledRadicalInverseSpecialized<2267>(a, perm);
    case 336:
        return ScrambledRadicalInverseSpecialized<2269>(a, perm);
    case 337:
        return ScrambledRadicalInverseSpecialized<2273>(a, perm);
    case 338:
        return ScrambledRadicalInverseSpecialized<2281>(a, perm);
    case 339:
        return ScrambledRadicalInverseSpecialized<2287>(a, perm);
    case 340:
        return ScrambledRadicalInverseSpecialized<2293>(a, perm);
    case 341:
        return ScrambledRadicalInverseSpecialized<2297>(a, perm);
    case 342:
        return ScrambledRadicalInverseSpecialized<2309>(a, perm);
    case 343:
        return ScrambledRadicalInverseSpecialized<2311>(a, perm);
    case 344:
        return ScrambledRadicalInverseSpecialized<2333>(a, perm);
    case 345:
        return ScrambledRadicalInverseSpecialized<2339>(a, perm);
    case 346:
        return ScrambledRadicalInverseSpecialized<2341>(a, perm);
    case 347:
        return ScrambledRadicalInverseSpecialized<2347>(a, perm);
    case 348:
        return ScrambledRadicalInverseSpecialized<2351>(a, perm);
    case 349:
        return ScrambledRadicalInverseSpecialized<2357>(a, perm);
    case 350:
        return ScrambledRadicalInverseSpecialized<2371>(a, perm);
    case 351:
        return ScrambledRadicalInverseSpecialized<2377>(a, perm);
    case 352:
        return ScrambledRadicalInverseSpecialized<2381>(a, perm);
    case 353:
        return ScrambledRadicalInverseSpecialized<2383>(a, perm);
    case 354:
        return ScrambledRadicalInverseSpecialized<2389>(a, perm);
    case 355:
        return ScrambledRadicalInverseSpecialized<2393>(a, perm);
    case 356:
        return ScrambledRadicalInverseSpecialized<2399>(a, perm);
    case 357:
        return ScrambledRadicalInverseSpecialized<2411>(a, perm);
    case 358:
        return ScrambledRadicalInverseSpecialized<2417>(a, perm);
    case 359:
        return ScrambledRadicalInverseSpecialized<2423>(a, perm);
    case 360:
        return ScrambledRadicalInverseSpecialized<2437>(a, perm);
    case 361:
        return ScrambledRadicalInverseSpecialized<2441>(a, perm);
    case 362:
        return ScrambledRadicalInverseSpecialized<2447>(a, perm);
    case 363:
        return ScrambledRadicalInverseSpecialized<2459>(a, perm);
    case 364:
        return ScrambledRadicalInverseSpecialized<2467>(a, perm);
    case 365:
        return ScrambledRadicalInverseSpecialized<2473>(a, perm);
    case 366:
        return ScrambledRadicalInverseSpecialized<2477>(a, perm);
    case 367:
        return ScrambledRadicalInverseSpecialized<2503>(a, perm);
    case 368:
        return ScrambledRadicalInverseSpecialized<2521>(a, perm);
    case 369:
        return ScrambledRadicalInverseSpecialized<2531>(a, perm);
    case 370:
        return ScrambledRadicalInverseSpecialized<2539>(a, perm);
    case 371:
        return ScrambledRadicalInverseSpecialized<2543>(a, perm);
    case 372:
        return ScrambledRadicalInverseSpecialized<2549>(a, perm);
    case 373:
        return ScrambledRadicalInverseSpecialized<2551>(a, perm);
    case 374:
        return ScrambledRadicalInverseSpecialized<2557>(a, perm);
    case 375:
        return ScrambledRadicalInverseSpecialized<2579>(a, perm);
    case 376:
        return ScrambledRadicalInverseSpecialized<2591>(a, perm);
    case 377:
        return ScrambledRadicalInverseSpecialized<2593>(a, perm);
    case 378:
        return ScrambledRadicalInverseSpecialized<2609>(a, perm);
    case 379:
        return ScrambledRadicalInverseSpecialized<2617>(a, perm);
    case 380:
        return ScrambledRadicalInverseSpecialized<2621>(a, perm);
    case 381:
        return ScrambledRadicalInverseSpecialized<2633>(a, perm);
    case 382:
        return ScrambledRadicalInverseSpecialized<2647>(a, perm);
    case 383:
        return ScrambledRadicalInverseSpecialized<2657>(a, perm);
    case 384:
        return ScrambledRadicalInverseSpecialized<2659>(a, perm);
    case 385:
        return ScrambledRadicalInverseSpecialized<2663>(a, perm);
    case 386:
        return ScrambledRadicalInverseSpecialized<2671>(a, perm);
    case 387:
        return ScrambledRadicalInverseSpecialized<2677>(a, perm);
    case 388:
        return ScrambledRadicalInverseSpecialized<2683>(a, perm);
    case 389:
        return ScrambledRadicalInverseSpecialized<2687>(a, perm);
    case 390:
        return ScrambledRadicalInverseSpecialized<2689>(a, perm);
    case 391:
        return ScrambledRadicalInverseSpecialized<2693>(a, perm);
    case 392:
        return ScrambledRadicalInverseSpecialized<2699>(a, perm);
    case 393:
        return ScrambledRadicalInverseSpecialized<2707>(a, perm);
    case 394:
        return ScrambledRadicalInverseSpecialized<2711>(a, perm);
    case 395:
        return ScrambledRadicalInverseSpecialized<2713>(a, perm);
    case 396:
        return ScrambledRadicalInverseSpecialized<2719>(a, perm);
    case 397:
        return ScrambledRadicalInverseSpecialized<2729>(a, perm);
    case 398:
        return ScrambledRadicalInverseSpecialized<2731>(a, perm);
    case 399:
        return ScrambledRadicalInverseSpecialized<2741>(a, perm);
    case 400:
        return ScrambledRadicalInverseSpecialized<2749>(a, perm);
    case 401:
        return ScrambledRadicalInverseSpecialized<2753>(a, perm);
    case 402:
        return ScrambledRadicalInverseSpecialized<2767>(a, perm);
    case 403:
        return ScrambledRadicalInverseSpecialized<2777>(a, perm);
    case 404:
        return ScrambledRadicalInverseSpecialized<2789>(a, perm);
    case 405:
        return ScrambledRadicalInverseSpecialized<2791>(a, perm);
    case 406:
        return ScrambledRadicalInverseSpecialized<2797>(a, perm);
    case 407:
        return ScrambledRadicalInverseSpecialized<2801>(a, perm);
    case 408:
        return ScrambledRadicalInverseSpecialized<2803>(a, perm);
    case 409:
        return ScrambledRadicalInverseSpecialized<2819>(a, perm);
    case 410:
        return ScrambledRadicalInverseSpecialized<2833>(a, perm);
    case 411:
        return ScrambledRadicalInverseSpecialized<2837>(a, perm);
    case 412:
        return ScrambledRadicalInverseSpecialized<2843>(a, perm);
    case 413:
        return ScrambledRadicalInverseSpecialized<2851>(a, perm);
    case 414:
        return ScrambledRadicalInverseSpecialized<2857>(a, perm);
    case 415:
        return ScrambledRadicalInverseSpecialized<2861>(a, perm);
    case 416:
        return ScrambledRadicalInverseSpecialized<2879>(a, perm);
    case 417:
        return ScrambledRadicalInverseSpecialized<2887>(a, perm);
    case 418:
        return ScrambledRadicalInverseSpecialized<2897>(a, perm);
    case 419:
        return ScrambledRadicalInverseSpecialized<2903>(a, perm);
    case 420:
        return ScrambledRadicalInverseSpecialized<2909>(a, perm);
    case 421:
        return ScrambledRadicalInverseSpecialized<2917>(a, perm);
    case 422:
        return ScrambledRadicalInverseSpecialized<2927>(a, perm);
    case 423:
        return ScrambledRadicalInverseSpecialized<2939>(a, perm);
    case 424:
        return ScrambledRadicalInverseSpecialized<2953>(a, perm);
    case 425:
        return ScrambledRadicalInverseSpecialized<2957>(a, perm);
    case 426:
        return ScrambledRadicalInverseSpecialized<2963>(a, perm);
    case 427:
        return ScrambledRadicalInverseSpecialized<2969>(a, perm);
    case 428:
        return ScrambledRadicalInverseSpecialized<2971>(a, perm);
    case 429:
        return ScrambledRadicalInverseSpecialized<2999>(a, perm);
    case 430:
        return ScrambledRadicalInverseSpecialized<3001>(a, perm);
    case 431:
        return ScrambledRadicalInverseSpecialized<3011>(a, perm);
    case 432:
        return ScrambledRadicalInverseSpecialized<3019>(a, perm);
    case 433:
        return ScrambledRadicalInverseSpecialized<3023>(a, perm);
    case 434:
        return ScrambledRadicalInverseSpecialized<3037>(a, perm);
    case 435:
        return ScrambledRadicalInverseSpecialized<3041>(a, perm);
    case 436:
        return ScrambledRadicalInverseSpecialized<3049>(a, perm);
    case 437:
        return ScrambledRadicalInverseSpecialized<3061>(a, perm);
    case 438:
        return ScrambledRadicalInverseSpecialized<3067>(a, perm);
    case 439:
        return ScrambledRadicalInverseSpecialized<3079>(a, perm);
    case 440:
        return ScrambledRadicalInverseSpecialized<3083>(a, perm);
    case 441:
        return ScrambledRadicalInverseSpecialized<3089>(a, perm);
    case 442:
        return ScrambledRadicalInverseSpecialized<3109>(a, perm);
    case 443:
        return ScrambledRadicalInverseSpecialized<3119>(a, perm);
    case 444:
        return ScrambledRadicalInverseSpecialized<3121>(a, perm);
    case 445:
        return ScrambledRadicalInverseSpecialized<3137>(a, perm);
    case 446:
        return ScrambledRadicalInverseSpecialized<3163>(a, perm);
    case 447:
        return ScrambledRadicalInverseSpecialized<3167>(a, perm);
    case 448:
        return ScrambledRadicalInverseSpecialized<3169>(a, perm);
    case 449:
        return ScrambledRadicalInverseSpecialized<3181>(a, perm);
    case 450:
        return ScrambledRadicalInverseSpecialized<3187>(a, perm);
    case 451:
        return ScrambledRadicalInverseSpecialized<3191>(a, perm);
    case 452:
        return ScrambledRadicalInverseSpecialized<3203>(a, perm);
    case 453:
        return ScrambledRadicalInverseSpecialized<3209>(a, perm);
    case 454:
        return ScrambledRadicalInverseSpecialized<3217>(a, perm);
    case 455:
        return ScrambledRadicalInverseSpecialized<3221>(a, perm);
    case 456:
        return ScrambledRadicalInverseSpecialized<3229>(a, perm);
    case 457:
        return ScrambledRadicalInverseSpecialized<3251>(a, perm);
    case 458:
        return ScrambledRadicalInverseSpecialized<3253>(a, perm);
    case 459:
        return ScrambledRadicalInverseSpecialized<3257>(a, perm);
    case 460:
        return ScrambledRadicalInverseSpecialized<3259>(a, perm);
    case 461:
        return ScrambledRadicalInverseSpecialized<3271>(a, perm);
    case 462:
        return ScrambledRadicalInverseSpecialized<3299>(a, perm);
    case 463:
        return ScrambledRadicalInverseSpecialized<3301>(a, perm);
    case 464:
        return ScrambledRadicalInverseSpecialized<3307>(a, perm);
    case 465:
        return ScrambledRadicalInverseSpecialized<3313>(a, perm);
    case 466:
        return ScrambledRadicalInverseSpecialized<3319>(a, perm);
    case 467:
        return ScrambledRadicalInverseSpecialized<3323>(a, perm);
    case 468:
        return ScrambledRadicalInverseSpecialized<3329>(a, perm);
    case 469:
        return ScrambledRadicalInverseSpecialized<3331>(a, perm);
    case 470:
        return ScrambledRadicalInverseSpecialized<3343>(a, perm);
    case 471:
        return ScrambledRadicalInverseSpecialized<3347>(a, perm);
    case 472:
        return ScrambledRadicalInverseSpecialized<3359>(a, perm);
    case 473:
        return ScrambledRadicalInverseSpecialized<3361>(a, perm);
    case 474:
        return ScrambledRadicalInverseSpecialized<3371>(a, perm);
    case 475:
        return ScrambledRadicalInverseSpecialized<3373>(a, perm);
    case 476:
        return ScrambledRadicalInverseSpecialized<3389>(a, perm);
    case 477:
        return ScrambledRadicalInverseSpecialized<3391>(a, perm);
    case 478:
        return ScrambledRadicalInverseSpecialized<3407>(a, perm);
    case 479:
        return ScrambledRadicalInverseSpecialized<3413>(a, perm);
    case 480:
        return ScrambledRadicalInverseSpecialized<3433>(a, perm);
    case 481:
        return ScrambledRadicalInverseSpecialized<3449>(a, perm);
    case 482:
        return ScrambledRadicalInverseSpecialized<3457>(a, perm);
    case 483:
        return ScrambledRadicalInverseSpecialized<3461>(a, perm);
    case 484:
        return ScrambledRadicalInverseSpecialized<3463>(a, perm);
    case 485:
        return ScrambledRadicalInverseSpecialized<3467>(a, perm);
    case 486:
        return ScrambledRadicalInverseSpecialized<3469>(a, perm);
    case 487:
        return ScrambledRadicalInverseSpecialized<3491>(a, perm);
    case 488:
        return ScrambledRadicalInverseSpecialized<3499>(a, perm);
    case 489:
        return ScrambledRadicalInverseSpecialized<3511>(a, perm);
    case 490:
        return ScrambledRadicalInverseSpecialized<3517>(a, perm);
    case 491:
        return ScrambledRadicalInverseSpecialized<3527>(a, perm);
    case 492:
        return ScrambledRadicalInverseSpecialized<3529>(a, perm);
    case 493:
        return ScrambledRadicalInverseSpecialized<3533>(a, perm);
    case 494:
        return ScrambledRadicalInverseSpecialized<3539>(a, perm);
    case 495:
        return ScrambledRadicalInverseSpecialized<3541>(a, perm);
    case 496:
        return ScrambledRadicalInverseSpecialized<3547>(a, perm);
    case 497:
        return ScrambledRadicalInverseSpecialized<3557>(a, perm);
    case 498:
        return ScrambledRadicalInverseSpecialized<3559>(a, perm);
    case 499:
        return ScrambledRadicalInverseSpecialized<3571>(a, perm);
    case 500:
        return ScrambledRadicalInverseSpecialized<3581>(a, perm);
    case 501:
        return ScrambledRadicalInverseSpecialized<3583>(a, perm);
    case 502:
        return ScrambledRadicalInverseSpecialized<3593>(a, perm);
    case 503:
        return ScrambledRadicalInverseSpecialized<3607>(a, perm);
    case 504:
        return ScrambledRadicalInverseSpecialized<3613>(a, perm);
    case 505:
        return ScrambledRadicalInverseSpecialized<3617>(a, perm);
    case 506:
        return ScrambledRadicalInverseSpecialized<3623>(a, perm);
    case 507:
        return ScrambledRadicalInverseSpecialized<3631>(a, perm);
    case 508:
        return ScrambledRadicalInverseSpecialized<3637>(a, perm);
    case 509:
        return ScrambledRadicalInverseSpecialized<3643>(a, perm);
    case 510:
        return ScrambledRadicalInverseSpecialized<3659>(a, perm);
    case 511:
        return ScrambledRadicalInverseSpecialized<3671>(a, perm);
    case 512:
        return ScrambledRadicalInverseSpecialized<3673>(a, perm);
    case 513:
        return ScrambledRadicalInverseSpecialized<3677>(a, perm);
    case 514:
        return ScrambledRadicalInverseSpecialized<3691>(a, perm);
    case 515:
        return ScrambledRadicalInverseSpecialized<3697>(a, perm);
    case 516:
        return ScrambledRadicalInverseSpecialized<3701>(a, perm);
    case 517:
        return ScrambledRadicalInverseSpecialized<3709>(a, perm);
    case 518:
        return ScrambledRadicalInverseSpecialized<3719>(a, perm);
    case 519:
        return ScrambledRadicalInverseSpecialized<3727>(a, perm);
    case 520:
        return ScrambledRadicalInverseSpecialized<3733>(a, perm);
    case 521:
        return ScrambledRadicalInverseSpecialized<3739>(a, perm);
    case 522:
        return ScrambledRadicalInverseSpecialized<3761>(a, perm);
    case 523:
        return ScrambledRadicalInverseSpecialized<3767>(a, perm);
    case 524:
        return ScrambledRadicalInverseSpecialized<3769>(a, perm);
    case 525:
        return ScrambledRadicalInverseSpecialized<3779>(a, perm);
    case 526:
        return ScrambledRadicalInverseSpecialized<3793>(a, perm);
    case 527:
        return ScrambledRadicalInverseSpecialized<3797>(a, perm);
    case 528:
        return ScrambledRadicalInverseSpecialized<3803>(a, perm);
    case 529:
        return ScrambledRadicalInverseSpecialized<3821>(a, perm);
    case 530:
        return ScrambledRadicalInverseSpecialized<3823>(a, perm);
    case 531:
        return ScrambledRadicalInverseSpecialized<3833>(a, perm);
    case 532:
        return ScrambledRadicalInverseSpecialized<3847>(a, perm);
    case 533:
        return ScrambledRadicalInverseSpecialized<3851>(a, perm);
    case 534:
        return ScrambledRadicalInverseSpecialized<3853>(a, perm);
    case 535:
        return ScrambledRadicalInverseSpecialized<3863>(a, perm);
    case 536:
        return ScrambledRadicalInverseSpecialized<3877>(a, perm);
    case 537:
        return ScrambledRadicalInverseSpecialized<3881>(a, perm);
    case 538:
        return ScrambledRadicalInverseSpecialized<3889>(a, perm);
    case 539:
        return ScrambledRadicalInverseSpecialized<3907>(a, perm);
    case 540:
        return ScrambledRadicalInverseSpecialized<3911>(a, perm);
    case 541:
        return ScrambledRadicalInverseSpecialized<3917>(a, perm);
    case 542:
        return ScrambledRadicalInverseSpecialized<3919>(a, perm);
    case 543:
        return ScrambledRadicalInverseSpecialized<3923>(a, perm);
    case 544:
        return ScrambledRadicalInverseSpecialized<3929>(a, perm);
    case 545:
        return ScrambledRadicalInverseSpecialized<3931>(a, perm);
    case 546:
        return ScrambledRadicalInverseSpecialized<3943>(a, perm);
    case 547:
        return ScrambledRadicalInverseSpecialized<3947>(a, perm);
    case 548:
        return ScrambledRadicalInverseSpecialized<3967>(a, perm);
    case 549:
        return ScrambledRadicalInverseSpecialized<3989>(a, perm);
    case 550:
        return ScrambledRadicalInverseSpecialized<4001>(a, perm);
    case 551:
        return ScrambledRadicalInverseSpecialized<4003>(a, perm);
    case 552:
        return ScrambledRadicalInverseSpecialized<4007>(a, perm);
    case 553:
        return ScrambledRadicalInverseSpecialized<4013>(a, perm);
    case 554:
        return ScrambledRadicalInverseSpecialized<4019>(a, perm);
    case 555:
        return ScrambledRadicalInverseSpecialized<4021>(a, perm);
    case 556:
        return ScrambledRadicalInverseSpecialized<4027>(a, perm);
    case 557:
        return ScrambledRadicalInverseSpecialized<4049>(a, perm);
    case 558:
        return ScrambledRadicalInverseSpecialized<4051>(a, perm);
    case 559:
        return ScrambledRadicalInverseSpecialized<4057>(a, perm);
    case 560:
        return ScrambledRadicalInverseSpecialized<4073>(a, perm);
    case 561:
        return ScrambledRadicalInverseSpecialized<4079>(a, perm);
    case 562:
        return ScrambledRadicalInverseSpecialized<4091>(a, perm);
    case 563:
        return ScrambledRadicalInverseSpecialized<4093>(a, perm);
    case 564:
        return ScrambledRadicalInverseSpecialized<4099>(a, perm);
    case 565:
        return ScrambledRadicalInverseSpecialized<4111>(a, perm);
    case 566:
        return ScrambledRadicalInverseSpecialized<4127>(a, perm);
    case 567:
        return ScrambledRadicalInverseSpecialized<4129>(a, perm);
    case 568:
        return ScrambledRadicalInverseSpecialized<4133>(a, perm);
    case 569:
        return ScrambledRadicalInverseSpecialized<4139>(a, perm);
    case 570:
        return ScrambledRadicalInverseSpecialized<4153>(a, perm);
    case 571:
        return ScrambledRadicalInverseSpecialized<4157>(a, perm);
    case 572:
        return ScrambledRadicalInverseSpecialized<4159>(a, perm);
    case 573:
        return ScrambledRadicalInverseSpecialized<4177>(a, perm);
    case 574:
        return ScrambledRadicalInverseSpecialized<4201>(a, perm);
    case 575:
        return ScrambledRadicalInverseSpecialized<4211>(a, perm);
    case 576:
        return ScrambledRadicalInverseSpecialized<4217>(a, perm);
    case 577:
        return ScrambledRadicalInverseSpecialized<4219>(a, perm);
    case 578:
        return ScrambledRadicalInverseSpecialized<4229>(a, perm);
    case 579:
        return ScrambledRadicalInverseSpecialized<4231>(a, perm);
    case 580:
        return ScrambledRadicalInverseSpecialized<4241>(a, perm);
    case 581:
        return ScrambledRadicalInverseSpecialized<4243>(a, perm);
    case 582:
        return ScrambledRadicalInverseSpecialized<4253>(a, perm);
    case 583:
        return ScrambledRadicalInverseSpecialized<4259>(a, perm);
    case 584:
        return ScrambledRadicalInverseSpecialized<4261>(a, perm);
    case 585:
        return ScrambledRadicalInverseSpecialized<4271>(a, perm);
    case 586:
        return ScrambledRadicalInverseSpecialized<4273>(a, perm);
    case 587:
        return ScrambledRadicalInverseSpecialized<4283>(a, perm);
    case 588:
        return ScrambledRadicalInverseSpecialized<4289>(a, perm);
    case 589:
        return ScrambledRadicalInverseSpecialized<4297>(a, perm);
    case 590:
        return ScrambledRadicalInverseSpecialized<4327>(a, perm);
    case 591:
        return ScrambledRadicalInverseSpecialized<4337>(a, perm);
    case 592:
        return ScrambledRadicalInverseSpecialized<4339>(a, perm);
    case 593:
        return ScrambledRadicalInverseSpecialized<4349>(a, perm);
    case 594:
        return ScrambledRadicalInverseSpecialized<4357>(a, perm);
    case 595:
        return ScrambledRadicalInverseSpecialized<4363>(a, perm);
    case 596:
        return ScrambledRadicalInverseSpecialized<4373>(a, perm);
    case 597:
        return ScrambledRadicalInverseSpecialized<4391>(a, perm);
    case 598:
        return ScrambledRadicalInverseSpecialized<4397>(a, perm);
    case 599:
        return ScrambledRadicalInverseSpecialized<4409>(a, perm);
    case 600:
        return ScrambledRadicalInverseSpecialized<4421>(a, perm);
    case 601:
        return ScrambledRadicalInverseSpecialized<4423>(a, perm);
    case 602:
        return ScrambledRadicalInverseSpecialized<4441>(a, perm);
    case 603:
        return ScrambledRadicalInverseSpecialized<4447>(a, perm);
    case 604:
        return ScrambledRadicalInverseSpecialized<4451>(a, perm);
    case 605:
        return ScrambledRadicalInverseSpecialized<4457>(a, perm);
    case 606:
        return ScrambledRadicalInverseSpecialized<4463>(a, perm);
    case 607:
        return ScrambledRadicalInverseSpecialized<4481>(a, perm);
    case 608:
        return ScrambledRadicalInverseSpecialized<4483>(a, perm);
    case 609:
        return ScrambledRadicalInverseSpecialized<4493>(a, perm);
    case 610:
        return ScrambledRadicalInverseSpecialized<4507>(a, perm);
    case 611:
        return ScrambledRadicalInverseSpecialized<4513>(a, perm);
    case 612:
        return ScrambledRadicalInverseSpecialized<4517>(a, perm);
    case 613:
        return ScrambledRadicalInverseSpecialized<4519>(a, perm);
    case 614:
        return ScrambledRadicalInverseSpecialized<4523>(a, perm);
    case 615:
        return ScrambledRadicalInverseSpecialized<4547>(a, perm);
    case 616:
        return ScrambledRadicalInverseSpecialized<4549>(a, perm);
    case 617:
        return ScrambledRadicalInverseSpecialized<4561>(a, perm);
    case 618:
        return ScrambledRadicalInverseSpecialized<4567>(a, perm);
    case 619:
        return ScrambledRadicalInverseSpecialized<4583>(a, perm);
    case 620:
        return ScrambledRadicalInverseSpecialized<4591>(a, perm);
    case 621:
        return ScrambledRadicalInverseSpecialized<4597>(a, perm);
    case 622:
        return ScrambledRadicalInverseSpecialized<4603>(a, perm);
    case 623:
        return ScrambledRadicalInverseSpecialized<4621>(a, perm);
    case 624:
        return ScrambledRadicalInverseSpecialized<4637>(a, perm);
    case 625:
        return ScrambledRadicalInverseSpecialized<4639>(a, perm);
    case 626:
        return ScrambledRadicalInverseSpecialized<4643>(a, perm);
    case 627:
        return ScrambledRadicalInverseSpecialized<4649>(a, perm);
    case 628:
        return ScrambledRadicalInverseSpecialized<4651>(a, perm);
    case 629:
        return ScrambledRadicalInverseSpecialized<4657>(a, perm);
    case 630:
        return ScrambledRadicalInverseSpecialized<4663>(a, perm);
    case 631:
        return ScrambledRadicalInverseSpecialized<4673>(a, perm);
    case 632:
        return ScrambledRadicalInverseSpecialized<4679>(a, perm);
    case 633:
        return ScrambledRadicalInverseSpecialized<4691>(a, perm);
    case 634:
        return ScrambledRadicalInverseSpecialized<4703>(a, perm);
    case 635:
        return ScrambledRadicalInverseSpecialized<4721>(a, perm);
    case 636:
        return ScrambledRadicalInverseSpecialized<4723>(a, perm);
    case 637:
        return ScrambledRadicalInverseSpecialized<4729>(a, perm);
    case 638:
        return ScrambledRadicalInverseSpecialized<4733>(a, perm);
    case 639:
        return ScrambledRadicalInverseSpecialized<4751>(a, perm);
    case 640:
        return ScrambledRadicalInverseSpecialized<4759>(a, perm);
    case 641:
        return ScrambledRadicalInverseSpecialized<4783>(a, perm);
    case 642:
        return ScrambledRadicalInverseSpecialized<4787>(a, perm);
    case 643:
        return ScrambledRadicalInverseSpecialized<4789>(a, perm);
    case 644:
        return ScrambledRadicalInverseSpecialized<4793>(a, perm);
    case 645:
        return ScrambledRadicalInverseSpecialized<4799>(a, perm);
    case 646:
        return ScrambledRadicalInverseSpecialized<4801>(a, perm);
    case 647:
        return ScrambledRadicalInverseSpecialized<4813>(a, perm);
    case 648:
        return ScrambledRadicalInverseSpecialized<4817>(a, perm);
    case 649:
        return ScrambledRadicalInverseSpecialized<4831>(a, perm);
    case 650:
        return ScrambledRadicalInverseSpecialized<4861>(a, perm);
    case 651:
        return ScrambledRadicalInverseSpecialized<4871>(a, perm);
    case 652:
        return ScrambledRadicalInverseSpecialized<4877>(a, perm);
    case 653:
        return ScrambledRadicalInverseSpecialized<4889>(a, perm);
    case 654:
        return ScrambledRadicalInverseSpecialized<4903>(a, perm);
    case 655:
        return ScrambledRadicalInverseSpecialized<4909>(a, perm);
    case 656:
        return ScrambledRadicalInverseSpecialized<4919>(a, perm);
    case 657:
        return ScrambledRadicalInverseSpecialized<4931>(a, perm);
    case 658:
        return ScrambledRadicalInverseSpecialized<4933>(a, perm);
    case 659:
        return ScrambledRadicalInverseSpecialized<4937>(a, perm);
    case 660:
        return ScrambledRadicalInverseSpecialized<4943>(a, perm);
    case 661:
        return ScrambledRadicalInverseSpecialized<4951>(a, perm);
    case 662:
        return ScrambledRadicalInverseSpecialized<4957>(a, perm);
    case 663:
        return ScrambledRadicalInverseSpecialized<4967>(a, perm);
    case 664:
        return ScrambledRadicalInverseSpecialized<4969>(a, perm);
    case 665:
        return ScrambledRadicalInverseSpecialized<4973>(a, perm);
    case 666:
        return ScrambledRadicalInverseSpecialized<4987>(a, perm);
    case 667:
        return ScrambledRadicalInverseSpecialized<4993>(a, perm);
    case 668:
        return ScrambledRadicalInverseSpecialized<4999>(a, perm);
    case 669:
        return ScrambledRadicalInverseSpecialized<5003>(a, perm);
    case 670:
        return ScrambledRadicalInverseSpecialized<5009>(a, perm);
    case 671:
        return ScrambledRadicalInverseSpecialized<5011>(a, perm);
    case 672:
        return ScrambledRadicalInverseSpecialized<5021>(a, perm);
    case 673:
        return ScrambledRadicalInverseSpecialized<5023>(a, perm);
    case 674:
        return ScrambledRadicalInverseSpecialized<5039>(a, perm);
    case 675:
        return ScrambledRadicalInverseSpecialized<5051>(a, perm);
    case 676:
        return ScrambledRadicalInverseSpecialized<5059>(a, perm);
    case 677:
        return ScrambledRadicalInverseSpecialized<5077>(a, perm);
    case 678:
        return ScrambledRadicalInverseSpecialized<5081>(a, perm);
    case 679:
        return ScrambledRadicalInverseSpecialized<5087>(a, perm);
    case 680:
        return ScrambledRadicalInverseSpecialized<5099>(a, perm);
    case 681:
        return ScrambledRadicalInverseSpecialized<5101>(a, perm);
    case 682:
        return ScrambledRadicalInverseSpecialized<5107>(a, perm);
    case 683:
        return ScrambledRadicalInverseSpecialized<5113>(a, perm);
    case 684:
        return ScrambledRadicalInverseSpecialized<5119>(a, perm);
    case 685:
        return ScrambledRadicalInverseSpecialized<5147>(a, perm);
    case 686:
        return ScrambledRadicalInverseSpecialized<5153>(a, perm);
    case 687:
        return ScrambledRadicalInverseSpecialized<5167>(a, perm);
    case 688:
        return ScrambledRadicalInverseSpecialized<5171>(a, perm);
    case 689:
        return ScrambledRadicalInverseSpecialized<5179>(a, perm);
    case 690:
        return ScrambledRadicalInverseSpecialized<5189>(a, perm);
    case 691:
        return ScrambledRadicalInverseSpecialized<5197>(a, perm);
    case 692:
        return ScrambledRadicalInverseSpecialized<5209>(a, perm);
    case 693:
        return ScrambledRadicalInverseSpecialized<5227>(a, perm);
    case 694:
        return ScrambledRadicalInverseSpecialized<5231>(a, perm);
    case 695:
        return ScrambledRadicalInverseSpecialized<5233>(a, perm);
    case 696:
        return ScrambledRadicalInverseSpecialized<5237>(a, perm);
    case 697:
        return ScrambledRadicalInverseSpecialized<5261>(a, perm);
    case 698:
        return ScrambledRadicalInverseSpecialized<5273>(a, perm);
    case 699:
        return ScrambledRadicalInverseSpecialized<5279>(a, perm);
    case 700:
        return ScrambledRadicalInverseSpecialized<5281>(a, perm);
    case 701:
        return ScrambledRadicalInverseSpecialized<5297>(a, perm);
    case 702:
        return ScrambledRadicalInverseSpecialized<5303>(a, perm);
    case 703:
        return ScrambledRadicalInverseSpecialized<5309>(a, perm);
    case 704:
        return ScrambledRadicalInverseSpecialized<5323>(a, perm);
    case 705:
        return ScrambledRadicalInverseSpecialized<5333>(a, perm);
    case 706:
        return ScrambledRadicalInverseSpecialized<5347>(a, perm);
    case 707:
        return ScrambledRadicalInverseSpecialized<5351>(a, perm);
    case 708:
        return ScrambledRadicalInverseSpecialized<5381>(a, perm);
    case 709:
        return ScrambledRadicalInverseSpecialized<5387>(a, perm);
    case 710:
        return ScrambledRadicalInverseSpecialized<5393>(a, perm);
    case 711:
        return ScrambledRadicalInverseSpecialized<5399>(a, perm);
    case 712:
        return ScrambledRadicalInverseSpecialized<5407>(a, perm);
    case 713:
        return ScrambledRadicalInverseSpecialized<5413>(a, perm);
    case 714:
        return ScrambledRadicalInverseSpecialized<5417>(a, perm);
    case 715:
        return ScrambledRadicalInverseSpecialized<5419>(a, perm);
    case 716:
        return ScrambledRadicalInverseSpecialized<5431>(a, perm);
    case 717:
        return ScrambledRadicalInverseSpecialized<5437>(a, perm);
    case 718:
        return ScrambledRadicalInverseSpecialized<5441>(a, perm);
    case 719:
        return ScrambledRadicalInverseSpecialized<5443>(a, perm);
    case 720:
        return ScrambledRadicalInverseSpecialized<5449>(a, perm);
    case 721:
        return ScrambledRadicalInverseSpecialized<5471>(a, perm);
    case 722:
        return ScrambledRadicalInverseSpecialized<5477>(a, perm);
    case 723:
        return ScrambledRadicalInverseSpecialized<5479>(a, perm);
    case 724:
        return ScrambledRadicalInverseSpecialized<5483>(a, perm);
    case 725:
        return ScrambledRadicalInverseSpecialized<5501>(a, perm);
    case 726:
        return ScrambledRadicalInverseSpecialized<5503>(a, perm);
    case 727:
        return ScrambledRadicalInverseSpecialized<5507>(a, perm);
    case 728:
        return ScrambledRadicalInverseSpecialized<5519>(a, perm);
    case 729:
        return ScrambledRadicalInverseSpecialized<5521>(a, perm);
    case 730:
        return ScrambledRadicalInverseSpecialized<5527>(a, perm);
    case 731:
        return ScrambledRadicalInverseSpecialized<5531>(a, perm);
    case 732:
        return ScrambledRadicalInverseSpecialized<5557>(a, perm);
    case 733:
        return ScrambledRadicalInverseSpecialized<5563>(a, perm);
    case 734:
        return ScrambledRadicalInverseSpecialized<5569>(a, perm);
    case 735:
        return ScrambledRadicalInverseSpecialized<5573>(a, perm);
    case 736:
        return ScrambledRadicalInverseSpecialized<5581>(a, perm);
    case 737:
        return ScrambledRadicalInverseSpecialized<5591>(a, perm);
    case 738:
        return ScrambledRadicalInverseSpecialized<5623>(a, perm);
    case 739:
        return ScrambledRadicalInverseSpecialized<5639>(a, perm);
    case 740:
        return ScrambledRadicalInverseSpecialized<5641>(a, perm);
    case 741:
        return ScrambledRadicalInverseSpecialized<5647>(a, perm);
    case 742:
        return ScrambledRadicalInverseSpecialized<5651>(a, perm);
    case 743:
        return ScrambledRadicalInverseSpecialized<5653>(a, perm);
    case 744:
        return ScrambledRadicalInverseSpecialized<5657>(a, perm);
    case 745:
        return ScrambledRadicalInverseSpecialized<5659>(a, perm);
    case 746:
        return ScrambledRadicalInverseSpecialized<5669>(a, perm);
    case 747:
        return ScrambledRadicalInverseSpecialized<5683>(a, perm);
    case 748:
        return ScrambledRadicalInverseSpecialized<5689>(a, perm);
    case 749:
        return ScrambledRadicalInverseSpecialized<5693>(a, perm);
    case 750:
        return ScrambledRadicalInverseSpecialized<5701>(a, perm);
    case 751:
        return ScrambledRadicalInverseSpecialized<5711>(a, perm);
    case 752:
        return ScrambledRadicalInverseSpecialized<5717>(a, perm);
    case 753:
        return ScrambledRadicalInverseSpecialized<5737>(a, perm);
    case 754:
        return ScrambledRadicalInverseSpecialized<5741>(a, perm);
    case 755:
        return ScrambledRadicalInverseSpecialized<5743>(a, perm);
    case 756:
        return ScrambledRadicalInverseSpecialized<5749>(a, perm);
    case 757:
        return ScrambledRadicalInverseSpecialized<5779>(a, perm);
    case 758:
        return ScrambledRadicalInverseSpecialized<5783>(a, perm);
    case 759:
        return ScrambledRadicalInverseSpecialized<5791>(a, perm);
    case 760:
        return ScrambledRadicalInverseSpecialized<5801>(a, perm);
    case 761:
        return ScrambledRadicalInverseSpecialized<5807>(a, perm);
    case 762:
        return ScrambledRadicalInverseSpecialized<5813>(a, perm);
    case 763:
        return ScrambledRadicalInverseSpecialized<5821>(a, perm);
    case 764:
        return ScrambledRadicalInverseSpecialized<5827>(a, perm);
    case 765:
        return ScrambledRadicalInverseSpecialized<5839>(a, perm);
    case 766:
        return ScrambledRadicalInverseSpecialized<5843>(a, perm);
    case 767:
        return ScrambledRadicalInverseSpecialized<5849>(a, perm);
    case 768:
        return ScrambledRadicalInverseSpecialized<5851>(a, perm);
    case 769:
        return ScrambledRadicalInverseSpecialized<5857>(a, perm);
    case 770:
        return ScrambledRadicalInverseSpecialized<5861>(a, perm);
    case 771:
        return ScrambledRadicalInverseSpecialized<5867>(a, perm);
    case 772:
        return ScrambledRadicalInverseSpecialized<5869>(a, perm);
    case 773:
        return ScrambledRadicalInverseSpecialized<5879>(a, perm);
    case 774:
        return ScrambledRadicalInverseSpecialized<5881>(a, perm);
    case 775:
        return ScrambledRadicalInverseSpecialized<5897>(a, perm);
    case 776:
        return ScrambledRadicalInverseSpecialized<5903>(a, perm);
    case 777:
        return ScrambledRadicalInverseSpecialized<5923>(a, perm);
    case 778:
        return ScrambledRadicalInverseSpecialized<5927>(a, perm);
    case 779:
        return ScrambledRadicalInverseSpecialized<5939>(a, perm);
    case 780:
        return ScrambledRadicalInverseSpecialized<5953>(a, perm);
    case 781:
        return ScrambledRadicalInverseSpecialized<5981>(a, perm);
    case 782:
        return ScrambledRadicalInverseSpecialized<5987>(a, perm);
    case 783:
        return ScrambledRadicalInverseSpecialized<6007>(a, perm);
    case 784:
        return ScrambledRadicalInverseSpecialized<6011>(a, perm);
    case 785:
        return ScrambledRadicalInverseSpecialized<6029>(a, perm);
    case 786:
        return ScrambledRadicalInverseSpecialized<6037>(a, perm);
    case 787:
        return ScrambledRadicalInverseSpecialized<6043>(a, perm);
    case 788:
        return ScrambledRadicalInverseSpecialized<6047>(a, perm);
    case 789:
        return ScrambledRadicalInverseSpecialized<6053>(a, perm);
    case 790:
        return ScrambledRadicalInverseSpecialized<6067>(a, perm);
    case 791:
        return ScrambledRadicalInverseSpecialized<6073>(a, perm);
    case 792:
        return ScrambledRadicalInverseSpecialized<6079>(a, perm);
    case 793:
        return ScrambledRadicalInverseSpecialized<6089>(a, perm);
    case 794:
        return ScrambledRadicalInverseSpecialized<6091>(a, perm);
    case 795:
        return ScrambledRadicalInverseSpecialized<6101>(a, perm);
    case 796:
        return ScrambledRadicalInverseSpecialized<6113>(a, perm);
    case 797:
        return ScrambledRadicalInverseSpecialized<6121>(a, perm);
    case 798:
        return ScrambledRadicalInverseSpecialized<6131>(a, perm);
    case 799:
        return ScrambledRadicalInverseSpecialized<6133>(a, perm);
    case 800:
        return ScrambledRadicalInverseSpecialized<6143>(a, perm);
    case 801:
        return ScrambledRadicalInverseSpecialized<6151>(a, perm);
    case 802:
        return ScrambledRadicalInverseSpecialized<6163>(a, perm);
    case 803:
        return ScrambledRadicalInverseSpecialized<6173>(a, perm);
    case 804:
        return ScrambledRadicalInverseSpecialized<6197>(a, perm);
    case 805:
        return ScrambledRadicalInverseSpecialized<6199>(a, perm);
    case 806:
        return ScrambledRadicalInverseSpecialized<6203>(a, perm);
    case 807:
        return ScrambledRadicalInverseSpecialized<6211>(a, perm);
    case 808:
        return ScrambledRadicalInverseSpecialized<6217>(a, perm);
    case 809:
        return ScrambledRadicalInverseSpecialized<6221>(a, perm);
    case 810:
        return ScrambledRadicalInverseSpecialized<6229>(a, perm);
    case 811:
        return ScrambledRadicalInverseSpecialized<6247>(a, perm);
    case 812:
        return ScrambledRadicalInverseSpecialized<6257>(a, perm);
    case 813:
        return ScrambledRadicalInverseSpecialized<6263>(a, perm);
    case 814:
        return ScrambledRadicalInverseSpecialized<6269>(a, perm);
    case 815:
        return ScrambledRadicalInverseSpecialized<6271>(a, perm);
    case 816:
        return ScrambledRadicalInverseSpecialized<6277>(a, perm);
    case 817:
        return ScrambledRadicalInverseSpecialized<6287>(a, perm);
    case 818:
        return ScrambledRadicalInverseSpecialized<6299>(a, perm);
    case 819:
        return ScrambledRadicalInverseSpecialized<6301>(a, perm);
    case 820:
        return ScrambledRadicalInverseSpecialized<6311>(a, perm);
    case 821:
        return ScrambledRadicalInverseSpecialized<6317>(a, perm);
    case 822:
        return ScrambledRadicalInverseSpecialized<6323>(a, perm);
    case 823:
        return ScrambledRadicalInverseSpecialized<6329>(a, perm);
    case 824:
        return ScrambledRadicalInverseSpecialized<6337>(a, perm);
    case 825:
        return ScrambledRadicalInverseSpecialized<6343>(a, perm);
    case 826:
        return ScrambledRadicalInverseSpecialized<6353>(a, perm);
    case 827:
        return ScrambledRadicalInverseSpecialized<6359>(a, perm);
    case 828:
        return ScrambledRadicalInverseSpecialized<6361>(a, perm);
    case 829:
        return ScrambledRadicalInverseSpecialized<6367>(a, perm);
    case 830:
        return ScrambledRadicalInverseSpecialized<6373>(a, perm);
    case 831:
        return ScrambledRadicalInverseSpecialized<6379>(a, perm);
    case 832:
        return ScrambledRadicalInverseSpecialized<6389>(a, perm);
    case 833:
        return ScrambledRadicalInverseSpecialized<6397>(a, perm);
    case 834:
        return ScrambledRadicalInverseSpecialized<6421>(a, perm);
    case 835:
        return ScrambledRadicalInverseSpecialized<6427>(a, perm);
    case 836:
        return ScrambledRadicalInverseSpecialized<6449>(a, perm);
    case 837:
        return ScrambledRadicalInverseSpecialized<6451>(a, perm);
    case 838:
        return ScrambledRadicalInverseSpecialized<6469>(a, perm);
    case 839:
        return ScrambledRadicalInverseSpecialized<6473>(a, perm);
    case 840:
        return ScrambledRadicalInverseSpecialized<6481>(a, perm);
    case 841:
        return ScrambledRadicalInverseSpecialized<6491>(a, perm);
    case 842:
        return ScrambledRadicalInverseSpecialized<6521>(a, perm);
    case 843:
        return ScrambledRadicalInverseSpecialized<6529>(a, perm);
    case 844:
        return ScrambledRadicalInverseSpecialized<6547>(a, perm);
    case 845:
        return ScrambledRadicalInverseSpecialized<6551>(a, perm);
    case 846:
        return ScrambledRadicalInverseSpecialized<6553>(a, perm);
    case 847:
        return ScrambledRadicalInverseSpecialized<6563>(a, perm);
    case 848:
        return ScrambledRadicalInverseSpecialized<6569>(a, perm);
    case 849:
        return ScrambledRadicalInverseSpecialized<6571>(a, perm);
    case 850:
        return ScrambledRadicalInverseSpecialized<6577>(a, perm);
    case 851:
        return ScrambledRadicalInverseSpecialized<6581>(a, perm);
    case 852:
        return ScrambledRadicalInverseSpecialized<6599>(a, perm);
    case 853:
        return ScrambledRadicalInverseSpecialized<6607>(a, perm);
    case 854:
        return ScrambledRadicalInverseSpecialized<6619>(a, perm);
    case 855:
        return ScrambledRadicalInverseSpecialized<6637>(a, perm);
    case 856:
        return ScrambledRadicalInverseSpecialized<6653>(a, perm);
    case 857:
        return ScrambledRadicalInverseSpecialized<6659>(a, perm);
    case 858:
        return ScrambledRadicalInverseSpecialized<6661>(a, perm);
    case 859:
        return ScrambledRadicalInverseSpecialized<6673>(a, perm);
    case 860:
        return ScrambledRadicalInverseSpecialized<6679>(a, perm);
    case 861:
        return ScrambledRadicalInverseSpecialized<6689>(a, perm);
    case 862:
        return ScrambledRadicalInverseSpecialized<6691>(a, perm);
    case 863:
        return ScrambledRadicalInverseSpecialized<6701>(a, perm);
    case 864:
        return ScrambledRadicalInverseSpecialized<6703>(a, perm);
    case 865:
        return ScrambledRadicalInverseSpecialized<6709>(a, perm);
    case 866:
        return ScrambledRadicalInverseSpecialized<6719>(a, perm);
    case 867:
        return ScrambledRadicalInverseSpecialized<6733>(a, perm);
    case 868:
        return ScrambledRadicalInverseSpecialized<6737>(a, perm);
    case 869:
        return ScrambledRadicalInverseSpecialized<6761>(a, perm);
    case 870:
        return ScrambledRadicalInverseSpecialized<6763>(a, perm);
    case 871:
        return ScrambledRadicalInverseSpecialized<6779>(a, perm);
    case 872:
        return ScrambledRadicalInverseSpecialized<6781>(a, perm);
    case 873:
        return ScrambledRadicalInverseSpecialized<6791>(a, perm);
    case 874:
        return ScrambledRadicalInverseSpecialized<6793>(a, perm);
    case 875:
        return ScrambledRadicalInverseSpecialized<6803>(a, perm);
    case 876:
        return ScrambledRadicalInverseSpecialized<6823>(a, perm);
    case 877:
        return ScrambledRadicalInverseSpecialized<6827>(a, perm);
    case 878:
        return ScrambledRadicalInverseSpecialized<6829>(a, perm);
    case 879:
        return ScrambledRadicalInverseSpecialized<6833>(a, perm);
    case 880:
        return ScrambledRadicalInverseSpecialized<6841>(a, perm);
    case 881:
        return ScrambledRadicalInverseSpecialized<6857>(a, perm);
    case 882:
        return ScrambledRadicalInverseSpecialized<6863>(a, perm);
    case 883:
        return ScrambledRadicalInverseSpecialized<6869>(a, perm);
    case 884:
        return ScrambledRadicalInverseSpecialized<6871>(a, perm);
    case 885:
        return ScrambledRadicalInverseSpecialized<6883>(a, perm);
    case 886:
        return ScrambledRadicalInverseSpecialized<6899>(a, perm);
    case 887:
        return ScrambledRadicalInverseSpecialized<6907>(a, perm);
    case 888:
        return ScrambledRadicalInverseSpecialized<6911>(a, perm);
    case 889:
        return ScrambledRadicalInverseSpecialized<6917>(a, perm);
    case 890:
        return ScrambledRadicalInverseSpecialized<6947>(a, perm);
    case 891:
        return ScrambledRadicalInverseSpecialized<6949>(a, perm);
    case 892:
        return ScrambledRadicalInverseSpecialized<6959>(a, perm);
    case 893:
        return ScrambledRadicalInverseSpecialized<6961>(a, perm);
    case 894:
        return ScrambledRadicalInverseSpecialized<6967>(a, perm);
    case 895:
        return ScrambledRadicalInverseSpecialized<6971>(a, perm);
    case 896:
        return ScrambledRadicalInverseSpecialized<6977>(a, perm);
    case 897:
        return ScrambledRadicalInverseSpecialized<6983>(a, perm);
    case 898:
        return ScrambledRadicalInverseSpecialized<6991>(a, perm);
    case 899:
        return ScrambledRadicalInverseSpecialized<6997>(a, perm);
    case 900:
        return ScrambledRadicalInverseSpecialized<7001>(a, perm);
    case 901:
        return ScrambledRadicalInverseSpecialized<7013>(a, perm);
    case 902:
        return ScrambledRadicalInverseSpecialized<7019>(a, perm);
    case 903:
        return ScrambledRadicalInverseSpecialized<7027>(a, perm);
    case 904:
        return ScrambledRadicalInverseSpecialized<7039>(a, perm);
    case 905:
        return ScrambledRadicalInverseSpecialized<7043>(a, perm);
    case 906:
        return ScrambledRadicalInverseSpecialized<7057>(a, perm);
    case 907:
        return ScrambledRadicalInverseSpecialized<7069>(a, perm);
    case 908:
        return ScrambledRadicalInverseSpecialized<7079>(a, perm);
    case 909:
        return ScrambledRadicalInverseSpecialized<7103>(a, perm);
    case 910:
        return ScrambledRadicalInverseSpecialized<7109>(a, perm);
    case 911:
        return ScrambledRadicalInverseSpecialized<7121>(a, perm);
    case 912:
        return ScrambledRadicalInverseSpecialized<7127>(a, perm);
    case 913:
        return ScrambledRadicalInverseSpecialized<7129>(a, perm);
    case 914:
        return ScrambledRadicalInverseSpecialized<7151>(a, perm);
    case 915:
        return ScrambledRadicalInverseSpecialized<7159>(a, perm);
    case 916:
        return ScrambledRadicalInverseSpecialized<7177>(a, perm);
    case 917:
        return ScrambledRadicalInverseSpecialized<7187>(a, perm);
    case 918:
        return ScrambledRadicalInverseSpecialized<7193>(a, perm);
    case 919:
        return ScrambledRadicalInverseSpecialized<7207>(a, perm);
    case 920:
        return ScrambledRadicalInverseSpecialized<7211>(a, perm);
    case 921:
        return ScrambledRadicalInverseSpecialized<7213>(a, perm);
    case 922:
        return ScrambledRadicalInverseSpecialized<7219>(a, perm);
    case 923:
        return ScrambledRadicalInverseSpecialized<7229>(a, perm);
    case 924:
        return ScrambledRadicalInverseSpecialized<7237>(a, perm);
    case 925:
        return ScrambledRadicalInverseSpecialized<7243>(a, perm);
    case 926:
        return ScrambledRadicalInverseSpecialized<7247>(a, perm);
    case 927:
        return ScrambledRadicalInverseSpecialized<7253>(a, perm);
    case 928:
        return ScrambledRadicalInverseSpecialized<7283>(a, perm);
    case 929:
        return ScrambledRadicalInverseSpecialized<7297>(a, perm);
    case 930:
        return ScrambledRadicalInverseSpecialized<7307>(a, perm);
    case 931:
        return ScrambledRadicalInverseSpecialized<7309>(a, perm);
    case 932:
        return ScrambledRadicalInverseSpecialized<7321>(a, perm);
    case 933:
        return ScrambledRadicalInverseSpecialized<7331>(a, perm);
    case 934:
        return ScrambledRadicalInverseSpecialized<7333>(a, perm);
    case 935:
        return ScrambledRadicalInverseSpecialized<7349>(a, perm);
    case 936:
        return ScrambledRadicalInverseSpecialized<7351>(a, perm);
    case 937:
        return ScrambledRadicalInverseSpecialized<7369>(a, perm);
    case 938:
        return ScrambledRadicalInverseSpecialized<7393>(a, perm);
    case 939:
        return ScrambledRadicalInverseSpecialized<7411>(a, perm);
    case 940:
        return ScrambledRadicalInverseSpecialized<7417>(a, perm);
    case 941:
        return ScrambledRadicalInverseSpecialized<7433>(a, perm);
    case 942:
        return ScrambledRadicalInverseSpecialized<7451>(a, perm);
    case 943:
        return ScrambledRadicalInverseSpecialized<7457>(a, perm);
    case 944:
        return ScrambledRadicalInverseSpecialized<7459>(a, perm);
    case 945:
        return ScrambledRadicalInverseSpecialized<7477>(a, perm);
    case 946:
        return ScrambledRadicalInverseSpecialized<7481>(a, perm);
    case 947:
        return ScrambledRadicalInverseSpecialized<7487>(a, perm);
    case 948:
        return ScrambledRadicalInverseSpecialized<7489>(a, perm);
    case 949:
        return ScrambledRadicalInverseSpecialized<7499>(a, perm);
    case 950:
        return ScrambledRadicalInverseSpecialized<7507>(a, perm);
    case 951:
        return ScrambledRadicalInverseSpecialized<7517>(a, perm);
    case 952:
        return ScrambledRadicalInverseSpecialized<7523>(a, perm);
    case 953:
        return ScrambledRadicalInverseSpecialized<7529>(a, perm);
    case 954:
        return ScrambledRadicalInverseSpecialized<7537>(a, perm);
    case 955:
        return ScrambledRadicalInverseSpecialized<7541>(a, perm);
    case 956:
        return ScrambledRadicalInverseSpecialized<7547>(a, perm);
    case 957:
        return ScrambledRadicalInverseSpecialized<7549>(a, perm);
    case 958:
        return ScrambledRadicalInverseSpecialized<7559>(a, perm);
    case 959:
        return ScrambledRadicalInverseSpecialized<7561>(a, perm);
    case 960:
        return ScrambledRadicalInverseSpecialized<7573>(a, perm);
    case 961:
        return ScrambledRadicalInverseSpecialized<7577>(a, perm);
    case 962:
        return ScrambledRadicalInverseSpecialized<7583>(a, perm);
    case 963:
        return ScrambledRadicalInverseSpecialized<7589>(a, perm);
    case 964:
        return ScrambledRadicalInverseSpecialized<7591>(a, perm);
    case 965:
        return ScrambledRadicalInverseSpecialized<7603>(a, perm);
    case 966:
        return ScrambledRadicalInverseSpecialized<7607>(a, perm);
    case 967:
        return ScrambledRadicalInverseSpecialized<7621>(a, perm);
    case 968:
        return ScrambledRadicalInverseSpecialized<7639>(a, perm);
    case 969:
        return ScrambledRadicalInverseSpecialized<7643>(a, perm);
    case 970:
        return ScrambledRadicalInverseSpecialized<7649>(a, perm);
    case 971:
        return ScrambledRadicalInverseSpecialized<7669>(a, perm);
    case 972:
        return ScrambledRadicalInverseSpecialized<7673>(a, perm);
    case 973:
        return ScrambledRadicalInverseSpecialized<7681>(a, perm);
    case 974:
        return ScrambledRadicalInverseSpecialized<7687>(a, perm);
    case 975:
        return ScrambledRadicalInverseSpecialized<7691>(a, perm);
    case 976:
        return ScrambledRadicalInverseSpecialized<7699>(a, perm);
    case 977:
        return ScrambledRadicalInverseSpecialized<7703>(a, perm);
    case 978:
        return ScrambledRadicalInverseSpecialized<7717>(a, perm);
    case 979:
        return ScrambledRadicalInverseSpecialized<7723>(a, perm);
    case 980:
        return ScrambledRadicalInverseSpecialized<7727>(a, perm);
    case 981:
        return ScrambledRadicalInverseSpecialized<7741>(a, perm);
    case 982:
        return ScrambledRadicalInverseSpecialized<7753>(a, perm);
    case 983:
        return ScrambledRadicalInverseSpecialized<7757>(a, perm);
    case 984:
        return ScrambledRadicalInverseSpecialized<7759>(a, perm);
    case 985:
        return ScrambledRadicalInverseSpecialized<7789>(a, perm);
    case 986:
        return ScrambledRadicalInverseSpecialized<7793>(a, perm);
    case 987:
        return ScrambledRadicalInverseSpecialized<7817>(a, perm);
    case 988:
        return ScrambledRadicalInverseSpecialized<7823>(a, perm);
    case 989:
        return ScrambledRadicalInverseSpecialized<7829>(a, perm);
    case 990:
        return ScrambledRadicalInverseSpecialized<7841>(a, perm);
    case 991:
        return ScrambledRadicalInverseSpecialized<7853>(a, perm);
    case 992:
        return ScrambledRadicalInverseSpecialized<7867>(a, perm);
    case 993:
        return ScrambledRadicalInverseSpecialized<7873>(a, perm);
    case 994:
        return ScrambledRadicalInverseSpecialized<7877>(a, perm);
    case 995:
        return ScrambledRadicalInverseSpecialized<7879>(a, perm);
    case 996:
        return ScrambledRadicalInverseSpecialized<7883>(a, perm);
    case 997:
        return ScrambledRadicalInverseSpecialized<7901>(a, perm);
    case 998:
        return ScrambledRadicalInverseSpecialized<7907>(a, perm);
    case 999:
        return ScrambledRadicalInverseSpecialized<7919>(a, perm);
    case 1000:
        return ScrambledRadicalInverseSpecialized<7927>(a, perm);
    case 1001:
        return ScrambledRadicalInverseSpecialized<7933>(a, perm);
    case 1002:
        return ScrambledRadicalInverseSpecialized<7937>(a, perm);
    case 1003:
        return ScrambledRadicalInverseSpecialized<7949>(a, perm);
    case 1004:
        return ScrambledRadicalInverseSpecialized<7951>(a, perm);
    case 1005:
        return ScrambledRadicalInverseSpecialized<7963>(a, perm);
    case 1006:
        return ScrambledRadicalInverseSpecialized<7993>(a, perm);
    case 1007:
        return ScrambledRadicalInverseSpecialized<8009>(a, perm);
    case 1008:
        return ScrambledRadicalInverseSpecialized<8011>(a, perm);
    case 1009:
        return ScrambledRadicalInverseSpecialized<8017>(a, perm);
    case 1010:
        return ScrambledRadicalInverseSpecialized<8039>(a, perm);
    case 1011:
        return ScrambledRadicalInverseSpecialized<8053>(a, perm);
    case 1012:
        return ScrambledRadicalInverseSpecialized<8059>(a, perm);
    case 1013:
        return ScrambledRadicalInverseSpecialized<8069>(a, perm);
    case 1014:
        return ScrambledRadicalInverseSpecialized<8081>(a, perm);
    case 1015:
        return ScrambledRadicalInverseSpecialized<8087>(a, perm);
    case 1016:
        return ScrambledRadicalInverseSpecialized<8089>(a, perm);
    case 1017:
        return ScrambledRadicalInverseSpecialized<8093>(a, perm);
    case 1018:
        return ScrambledRadicalInverseSpecialized<8101>(a, perm);
    case 1019:
        return ScrambledRadicalInverseSpecialized<8111>(a, perm);
    case 1020:
        return ScrambledRadicalInverseSpecialized<8117>(a, perm);
    case 1021:
        return ScrambledRadicalInverseSpecialized<8123>(a, perm);
    case 1022:
        return ScrambledRadicalInverseSpecialized<8147>(a, perm);
    case 1023:
        return ScrambledRadicalInverseSpecialized<8161>(a, perm);
    default:
        LOG_FATAL("Base %d is >= 1024, the limit of ScrambledRadicalInverse", baseIndex);
        return 0;
    }
}

Float ScrambledRadicalInverse(int baseIndex, uint64_t a, uint32_t seed) {
    switch (baseIndex) {
    case 0:
        return ScrambledRadicalInverseSpecialized<2>(a, seed);
    case 1:
        return ScrambledRadicalInverseSpecialized<3>(a, seed);
    case 2:
        return ScrambledRadicalInverseSpecialized<5>(a, seed);
    case 3:
        return ScrambledRadicalInverseSpecialized<7>(a, seed);
    // Remainder of cases for _ScrambledRadicalInverse()_
    case 4:
        return ScrambledRadicalInverseSpecialized<11>(a, seed);
    case 5:
        return ScrambledRadicalInverseSpecialized<13>(a, seed);
    case 6:
        return ScrambledRadicalInverseSpecialized<17>(a, seed);
    case 7:
        return ScrambledRadicalInverseSpecialized<19>(a, seed);
    case 8:
        return ScrambledRadicalInverseSpecialized<23>(a, seed);
    case 9:
        return ScrambledRadicalInverseSpecialized<29>(a, seed);
    case 10:
        return ScrambledRadicalInverseSpecialized<31>(a, seed);
    case 11:
        return ScrambledRadicalInverseSpecialized<37>(a, seed);
    case 12:
        return ScrambledRadicalInverseSpecialized<41>(a, seed);
    case 13:
        return ScrambledRadicalInverseSpecialized<43>(a, seed);
    case 14:
        return ScrambledRadicalInverseSpecialized<47>(a, seed);
    case 15:
        return ScrambledRadicalInverseSpecialized<53>(a, seed);
    case 16:
        return ScrambledRadicalInverseSpecialized<59>(a, seed);
    case 17:
        return ScrambledRadicalInverseSpecialized<61>(a, seed);
    case 18:
        return ScrambledRadicalInverseSpecialized<67>(a, seed);
    case 19:
        return ScrambledRadicalInverseSpecialized<71>(a, seed);
    case 20:
        return ScrambledRadicalInverseSpecialized<73>(a, seed);
    case 21:
        return ScrambledRadicalInverseSpecialized<79>(a, seed);
    case 22:
        return ScrambledRadicalInverseSpecialized<83>(a, seed);
    case 23:
        return ScrambledRadicalInverseSpecialized<89>(a, seed);
    case 24:
        return ScrambledRadicalInverseSpecialized<97>(a, seed);
    case 25:
        return ScrambledRadicalInverseSpecialized<101>(a, seed);
    case 26:
        return ScrambledRadicalInverseSpecialized<103>(a, seed);
    case 27:
        return ScrambledRadicalInverseSpecialized<107>(a, seed);
    case 28:
        return ScrambledRadicalInverseSpecialized<109>(a, seed);
    case 29:
        return ScrambledRadicalInverseSpecialized<113>(a, seed);
    case 30:
        return ScrambledRadicalInverseSpecialized<127>(a, seed);
    case 31:
        return ScrambledRadicalInverseSpecialized<131>(a, seed);
    case 32:
        return ScrambledRadicalInverseSpecialized<137>(a, seed);
    case 33:
        return ScrambledRadicalInverseSpecialized<139>(a, seed);
    case 34:
        return ScrambledRadicalInverseSpecialized<149>(a, seed);
    case 35:
        return ScrambledRadicalInverseSpecialized<151>(a, seed);
    case 36:
        return ScrambledRadicalInverseSpecialized<157>(a, seed);
    case 37:
        return ScrambledRadicalInverseSpecialized<163>(a, seed);
    case 38:
        return ScrambledRadicalInverseSpecialized<167>(a, seed);
    case 39:
        return ScrambledRadicalInverseSpecialized<173>(a, seed);
    case 40:
        return ScrambledRadicalInverseSpecialized<179>(a, seed);
    case 41:
        return ScrambledRadicalInverseSpecialized<181>(a, seed);
    case 42:
        return ScrambledRadicalInverseSpecialized<191>(a, seed);
    case 43:
        return ScrambledRadicalInverseSpecialized<193>(a, seed);
    case 44:
        return ScrambledRadicalInverseSpecialized<197>(a, seed);
    case 45:
        return ScrambledRadicalInverseSpecialized<199>(a, seed);
    case 46:
        return ScrambledRadicalInverseSpecialized<211>(a, seed);
    case 47:
        return ScrambledRadicalInverseSpecialized<223>(a, seed);
    case 48:
        return ScrambledRadicalInverseSpecialized<227>(a, seed);
    case 49:
        return ScrambledRadicalInverseSpecialized<229>(a, seed);
    case 50:
        return ScrambledRadicalInverseSpecialized<233>(a, seed);
    case 51:
        return ScrambledRadicalInverseSpecialized<239>(a, seed);
    case 52:
        return ScrambledRadicalInverseSpecialized<241>(a, seed);
    case 53:
        return ScrambledRadicalInverseSpecialized<251>(a, seed);
    case 54:
        return ScrambledRadicalInverseSpecialized<257>(a, seed);
    case 55:
        return ScrambledRadicalInverseSpecialized<263>(a, seed);
    case 56:
        return ScrambledRadicalInverseSpecialized<269>(a, seed);
    case 57:
        return ScrambledRadicalInverseSpecialized<271>(a, seed);
    case 58:
        return ScrambledRadicalInverseSpecialized<277>(a, seed);
    case 59:
        return ScrambledRadicalInverseSpecialized<281>(a, seed);
    case 60:
        return ScrambledRadicalInverseSpecialized<283>(a, seed);
    case 61:
        return ScrambledRadicalInverseSpecialized<293>(a, seed);
    case 62:
        return ScrambledRadicalInverseSpecialized<307>(a, seed);
    case 63:
        return ScrambledRadicalInverseSpecialized<311>(a, seed);
    case 64:
        return ScrambledRadicalInverseSpecialized<313>(a, seed);
    case 65:
        return ScrambledRadicalInverseSpecialized<317>(a, seed);
    case 66:
        return ScrambledRadicalInverseSpecialized<331>(a, seed);
    case 67:
        return ScrambledRadicalInverseSpecialized<337>(a, seed);
    case 68:
        return ScrambledRadicalInverseSpecialized<347>(a, seed);
    case 69:
        return ScrambledRadicalInverseSpecialized<349>(a, seed);
    case 70:
        return ScrambledRadicalInverseSpecialized<353>(a, seed);
    case 71:
        return ScrambledRadicalInverseSpecialized<359>(a, seed);
    case 72:
        return ScrambledRadicalInverseSpecialized<367>(a, seed);
    case 73:
        return ScrambledRadicalInverseSpecialized<373>(a, seed);
    case 74:
        return ScrambledRadicalInverseSpecialized<379>(a, seed);
    case 75:
        return ScrambledRadicalInverseSpecialized<383>(a, seed);
    case 76:
        return ScrambledRadicalInverseSpecialized<389>(a, seed);
    case 77:
        return ScrambledRadicalInverseSpecialized<397>(a, seed);
    case 78:
        return ScrambledRadicalInverseSpecialized<401>(a, seed);
    case 79:
        return ScrambledRadicalInverseSpecialized<409>(a, seed);
    case 80:
        return ScrambledRadicalInverseSpecialized<419>(a, seed);
    case 81:
        return ScrambledRadicalInverseSpecialized<421>(a, seed);
    case 82:
        return ScrambledRadicalInverseSpecialized<431>(a, seed);
    case 83:
        return ScrambledRadicalInverseSpecialized<433>(a, seed);
    case 84:
        return ScrambledRadicalInverseSpecialized<439>(a, seed);
    case 85:
        return ScrambledRadicalInverseSpecialized<443>(a, seed);
    case 86:
        return ScrambledRadicalInverseSpecialized<449>(a, seed);
    case 87:
        return ScrambledRadicalInverseSpecialized<457>(a, seed);
    case 88:
        return ScrambledRadicalInverseSpecialized<461>(a, seed);
    case 89:
        return ScrambledRadicalInverseSpecialized<463>(a, seed);
    case 90:
        return ScrambledRadicalInverseSpecialized<467>(a, seed);
    case 91:
        return ScrambledRadicalInverseSpecialized<479>(a, seed);
    case 92:
        return ScrambledRadicalInverseSpecialized<487>(a, seed);
    case 93:
        return ScrambledRadicalInverseSpecialized<491>(a, seed);
    case 94:
        return ScrambledRadicalInverseSpecialized<499>(a, seed);
    case 95:
        return ScrambledRadicalInverseSpecialized<503>(a, seed);
    case 96:
        return ScrambledRadicalInverseSpecialized<509>(a, seed);
    case 97:
        return ScrambledRadicalInverseSpecialized<521>(a, seed);
    case 98:
        return ScrambledRadicalInverseSpecialized<523>(a, seed);
    case 99:
        return ScrambledRadicalInverseSpecialized<541>(a, seed);
    case 100:
        return ScrambledRadicalInverseSpecialized<547>(a, seed);
    case 101:
        return ScrambledRadicalInverseSpecialized<557>(a, seed);
    case 102:
        return ScrambledRadicalInverseSpecialized<563>(a, seed);
    case 103:
        return ScrambledRadicalInverseSpecialized<569>(a, seed);
    case 104:
        return ScrambledRadicalInverseSpecialized<571>(a, seed);
    case 105:
        return ScrambledRadicalInverseSpecialized<577>(a, seed);
    case 106:
        return ScrambledRadicalInverseSpecialized<587>(a, seed);
    case 107:
        return ScrambledRadicalInverseSpecialized<593>(a, seed);
    case 108:
        return ScrambledRadicalInverseSpecialized<599>(a, seed);
    case 109:
        return ScrambledRadicalInverseSpecialized<601>(a, seed);
    case 110:
        return ScrambledRadicalInverseSpecialized<607>(a, seed);
    case 111:
        return ScrambledRadicalInverseSpecialized<613>(a, seed);
    case 112:
        return ScrambledRadicalInverseSpecialized<617>(a, seed);
    case 113:
        return ScrambledRadicalInverseSpecialized<619>(a, seed);
    case 114:
        return ScrambledRadicalInverseSpecialized<631>(a, seed);
    case 115:
        return ScrambledRadicalInverseSpecialized<641>(a, seed);
    case 116:
        return ScrambledRadicalInverseSpecialized<643>(a, seed);
    case 117:
        return ScrambledRadicalInverseSpecialized<647>(a, seed);
    case 118:
        return ScrambledRadicalInverseSpecialized<653>(a, seed);
    case 119:
        return ScrambledRadicalInverseSpecialized<659>(a, seed);
    case 120:
        return ScrambledRadicalInverseSpecialized<661>(a, seed);
    case 121:
        return ScrambledRadicalInverseSpecialized<673>(a, seed);
    case 122:
        return ScrambledRadicalInverseSpecialized<677>(a, seed);
    case 123:
        return ScrambledRadicalInverseSpecialized<683>(a, seed);
    case 124:
        return ScrambledRadicalInverseSpecialized<691>(a, seed);
    case 125:
        return ScrambledRadicalInverseSpecialized<701>(a, seed);
    case 126:
        return ScrambledRadicalInverseSpecialized<709>(a, seed);
    case 127:
        return ScrambledRadicalInverseSpecialized<719>(a, seed);
    case 128:
        return ScrambledRadicalInverseSpecialized<727>(a, seed);
    case 129:
        return ScrambledRadicalInverseSpecialized<733>(a, seed);
    case 130:
        return ScrambledRadicalInverseSpecialized<739>(a, seed);
    case 131:
        return ScrambledRadicalInverseSpecialized<743>(a, seed);
    case 132:
        return ScrambledRadicalInverseSpecialized<751>(a, seed);
    case 133:
        return ScrambledRadicalInverseSpecialized<757>(a, seed);
    case 134:
        return ScrambledRadicalInverseSpecialized<761>(a, seed);
    case 135:
        return ScrambledRadicalInverseSpecialized<769>(a, seed);
    case 136:
        return ScrambledRadicalInverseSpecialized<773>(a, seed);
    case 137:
        return ScrambledRadicalInverseSpecialized<787>(a, seed);
    case 138:
        return ScrambledRadicalInverseSpecialized<797>(a, seed);
    case 139:
        return ScrambledRadicalInverseSpecialized<809>(a, seed);
    case 140:
        return ScrambledRadicalInverseSpecialized<811>(a, seed);
    case 141:
        return ScrambledRadicalInverseSpecialized<821>(a, seed);
    case 142:
        return ScrambledRadicalInverseSpecialized<823>(a, seed);
    case 143:
        return ScrambledRadicalInverseSpecialized<827>(a, seed);
    case 144:
        return ScrambledRadicalInverseSpecialized<829>(a, seed);
    case 145:
        return ScrambledRadicalInverseSpecialized<839>(a, seed);
    case 146:
        return ScrambledRadicalInverseSpecialized<853>(a, seed);
    case 147:
        return ScrambledRadicalInverseSpecialized<857>(a, seed);
    case 148:
        return ScrambledRadicalInverseSpecialized<859>(a, seed);
    case 149:
        return ScrambledRadicalInverseSpecialized<863>(a, seed);
    case 150:
        return ScrambledRadicalInverseSpecialized<877>(a, seed);
    case 151:
        return ScrambledRadicalInverseSpecialized<881>(a, seed);
    case 152:
        return ScrambledRadicalInverseSpecialized<883>(a, seed);
    case 153:
        return ScrambledRadicalInverseSpecialized<887>(a, seed);
    case 154:
        return ScrambledRadicalInverseSpecialized<907>(a, seed);
    case 155:
        return ScrambledRadicalInverseSpecialized<911>(a, seed);
    case 156:
        return ScrambledRadicalInverseSpecialized<919>(a, seed);
    case 157:
        return ScrambledRadicalInverseSpecialized<929>(a, seed);
    case 158:
        return ScrambledRadicalInverseSpecialized<937>(a, seed);
    case 159:
        return ScrambledRadicalInverseSpecialized<941>(a, seed);
    case 160:
        return ScrambledRadicalInverseSpecialized<947>(a, seed);
    case 161:
        return ScrambledRadicalInverseSpecialized<953>(a, seed);
    case 162:
        return ScrambledRadicalInverseSpecialized<967>(a, seed);
    case 163:
        return ScrambledRadicalInverseSpecialized<971>(a, seed);
    case 164:
        return ScrambledRadicalInverseSpecialized<977>(a, seed);
    case 165:
        return ScrambledRadicalInverseSpecialized<983>(a, seed);
    case 166:
        return ScrambledRadicalInverseSpecialized<991>(a, seed);
    case 167:
        return ScrambledRadicalInverseSpecialized<997>(a, seed);
    case 168:
        return ScrambledRadicalInverseSpecialized<1009>(a, seed);
    case 169:
        return ScrambledRadicalInverseSpecialized<1013>(a, seed);
    case 170:
        return ScrambledRadicalInverseSpecialized<1019>(a, seed);
    case 171:
        return ScrambledRadicalInverseSpecialized<1021>(a, seed);
    case 172:
        return ScrambledRadicalInverseSpecialized<1031>(a, seed);
    case 173:
        return ScrambledRadicalInverseSpecialized<1033>(a, seed);
    case 174:
        return ScrambledRadicalInverseSpecialized<1039>(a, seed);
    case 175:
        return ScrambledRadicalInverseSpecialized<1049>(a, seed);
    case 176:
        return ScrambledRadicalInverseSpecialized<1051>(a, seed);
    case 177:
        return ScrambledRadicalInverseSpecialized<1061>(a, seed);
    case 178:
        return ScrambledRadicalInverseSpecialized<1063>(a, seed);
    case 179:
        return ScrambledRadicalInverseSpecialized<1069>(a, seed);
    case 180:
        return ScrambledRadicalInverseSpecialized<1087>(a, seed);
    case 181:
        return ScrambledRadicalInverseSpecialized<1091>(a, seed);
    case 182:
        return ScrambledRadicalInverseSpecialized<1093>(a, seed);
    case 183:
        return ScrambledRadicalInverseSpecialized<1097>(a, seed);
    case 184:
        return ScrambledRadicalInverseSpecialized<1103>(a, seed);
    case 185:
        return ScrambledRadicalInverseSpecialized<1109>(a, seed);
    case 186:
        return ScrambledRadicalInverseSpecialized<1117>(a, seed);
    case 187:
        return ScrambledRadicalInverseSpecialized<1123>(a, seed);
    case 188:
        return ScrambledRadicalInverseSpecialized<1129>(a, seed);
    case 189:
        return ScrambledRadicalInverseSpecialized<1151>(a, seed);
    case 190:
        return ScrambledRadicalInverseSpecialized<1153>(a, seed);
    case 191:
        return ScrambledRadicalInverseSpecialized<1163>(a, seed);
    case 192:
        return ScrambledRadicalInverseSpecialized<1171>(a, seed);
    case 193:
        return ScrambledRadicalInverseSpecialized<1181>(a, seed);
    case 194:
        return ScrambledRadicalInverseSpecialized<1187>(a, seed);
    case 195:
        return ScrambledRadicalInverseSpecialized<1193>(a, seed);
    case 196:
        return ScrambledRadicalInverseSpecialized<1201>(a, seed);
    case 197:
        return ScrambledRadicalInverseSpecialized<1213>(a, seed);
    case 198:
        return ScrambledRadicalInverseSpecialized<1217>(a, seed);
    case 199:
        return ScrambledRadicalInverseSpecialized<1223>(a, seed);
    case 200:
        return ScrambledRadicalInverseSpecialized<1229>(a, seed);
    case 201:
        return ScrambledRadicalInverseSpecialized<1231>(a, seed);
    case 202:
        return ScrambledRadicalInverseSpecialized<1237>(a, seed);
    case 203:
        return ScrambledRadicalInverseSpecialized<1249>(a, seed);
    case 204:
        return ScrambledRadicalInverseSpecialized<1259>(a, seed);
    case 205:
        return ScrambledRadicalInverseSpecialized<1277>(a, seed);
    case 206:
        return ScrambledRadicalInverseSpecialized<1279>(a, seed);
    case 207:
        return ScrambledRadicalInverseSpecialized<1283>(a, seed);
    case 208:
        return ScrambledRadicalInverseSpecialized<1289>(a, seed);
    case 209:
        return ScrambledRadicalInverseSpecialized<1291>(a, seed);
    case 210:
        return ScrambledRadicalInverseSpecialized<1297>(a, seed);
    case 211:
        return ScrambledRadicalInverseSpecialized<1301>(a, seed);
    case 212:
        return ScrambledRadicalInverseSpecialized<1303>(a, seed);
    case 213:
        return ScrambledRadicalInverseSpecialized<1307>(a, seed);
    case 214:
        return ScrambledRadicalInverseSpecialized<1319>(a, seed);
    case 215:
        return ScrambledRadicalInverseSpecialized<1321>(a, seed);
    case 216:
        return ScrambledRadicalInverseSpecialized<1327>(a, seed);
    case 217:
        return ScrambledRadicalInverseSpecialized<1361>(a, seed);
    case 218:
        return ScrambledRadicalInverseSpecialized<1367>(a, seed);
    case 219:
        return ScrambledRadicalInverseSpecialized<1373>(a, seed);
    case 220:
        return ScrambledRadicalInverseSpecialized<1381>(a, seed);
    case 221:
        return ScrambledRadicalInverseSpecialized<1399>(a, seed);
    case 222:
        return ScrambledRadicalInverseSpecialized<1409>(a, seed);
    case 223:
        return ScrambledRadicalInverseSpecialized<1423>(a, seed);
    case 224:
        return ScrambledRadicalInverseSpecialized<1427>(a, seed);
    case 225:
        return ScrambledRadicalInverseSpecialized<1429>(a, seed);
    case 226:
        return ScrambledRadicalInverseSpecialized<1433>(a, seed);
    case 227:
        return ScrambledRadicalInverseSpecialized<1439>(a, seed);
    case 228:
        return ScrambledRadicalInverseSpecialized<1447>(a, seed);
    case 229:
        return ScrambledRadicalInverseSpecialized<1451>(a, seed);
    case 230:
        return ScrambledRadicalInverseSpecialized<1453>(a, seed);
    case 231:
        return ScrambledRadicalInverseSpecialized<1459>(a, seed);
    case 232:
        return ScrambledRadicalInverseSpecialized<1471>(a, seed);
    case 233:
        return ScrambledRadicalInverseSpecialized<1481>(a, seed);
    case 234:
        return ScrambledRadicalInverseSpecialized<1483>(a, seed);
    case 235:
        return ScrambledRadicalInverseSpecialized<1487>(a, seed);
    case 236:
        return ScrambledRadicalInverseSpecialized<1489>(a, seed);
    case 237:
        return ScrambledRadicalInverseSpecialized<1493>(a, seed);
    case 238:
        return ScrambledRadicalInverseSpecialized<1499>(a, seed);
    case 239:
        return ScrambledRadicalInverseSpecialized<1511>(a, seed);
    case 240:
        return ScrambledRadicalInverseSpecialized<1523>(a, seed);
    case 241:
        return ScrambledRadicalInverseSpecialized<1531>(a, seed);
    case 242:
        return ScrambledRadicalInverseSpecialized<1543>(a, seed);
    case 243:
        return ScrambledRadicalInverseSpecialized<1549>(a, seed);
    case 244:
        return ScrambledRadicalInverseSpecialized<1553>(a, seed);
    case 245:
        return ScrambledRadicalInverseSpecialized<1559>(a, seed);
    case 246:
        return ScrambledRadicalInverseSpecialized<1567>(a, seed);
    case 247:
        return ScrambledRadicalInverseSpecialized<1571>(a, seed);
    case 248:
        return ScrambledRadicalInverseSpecialized<1579>(a, seed);
    case 249:
        return ScrambledRadicalInverseSpecialized<1583>(a, seed);
    case 250:
        return ScrambledRadicalInverseSpecialized<1597>(a, seed);
    case 251:
        return ScrambledRadicalInverseSpecialized<1601>(a, seed);
    case 252:
        return ScrambledRadicalInverseSpecialized<1607>(a, seed);
    case 253:
        return ScrambledRadicalInverseSpecialized<1609>(a, seed);
    case 254:
        return ScrambledRadicalInverseSpecialized<1613>(a, seed);
    case 255:
        return ScrambledRadicalInverseSpecialized<1619>(a, seed);
    case 256:
        return ScrambledRadicalInverseSpecialized<1621>(a, seed);
    case 257:
        return ScrambledRadicalInverseSpecialized<1627>(a, seed);
    case 258:
        return ScrambledRadicalInverseSpecialized<1637>(a, seed);
    case 259:
        return ScrambledRadicalInverseSpecialized<1657>(a, seed);
    case 260:
        return ScrambledRadicalInverseSpecialized<1663>(a, seed);
    case 261:
        return ScrambledRadicalInverseSpecialized<1667>(a, seed);
    case 262:
        return ScrambledRadicalInverseSpecialized<1669>(a, seed);
    case 263:
        return ScrambledRadicalInverseSpecialized<1693>(a, seed);
    case 264:
        return ScrambledRadicalInverseSpecialized<1697>(a, seed);
    case 265:
        return ScrambledRadicalInverseSpecialized<1699>(a, seed);
    case 266:
        return ScrambledRadicalInverseSpecialized<1709>(a, seed);
    case 267:
        return ScrambledRadicalInverseSpecialized<1721>(a, seed);
    case 268:
        return ScrambledRadicalInverseSpecialized<1723>(a, seed);
    case 269:
        return ScrambledRadicalInverseSpecialized<1733>(a, seed);
    case 270:
        return ScrambledRadicalInverseSpecialized<1741>(a, seed);
    case 271:
        return ScrambledRadicalInverseSpecialized<1747>(a, seed);
    case 272:
        return ScrambledRadicalInverseSpecialized<1753>(a, seed);
    case 273:
        return ScrambledRadicalInverseSpecialized<1759>(a, seed);
    case 274:
        return ScrambledRadicalInverseSpecialized<1777>(a, seed);
    case 275:
        return ScrambledRadicalInverseSpecialized<1783>(a, seed);
    case 276:
        return ScrambledRadicalInverseSpecialized<1787>(a, seed);
    case 277:
        return ScrambledRadicalInverseSpecialized<1789>(a, seed);
    case 278:
        return ScrambledRadicalInverseSpecialized<1801>(a, seed);
    case 279:
        return ScrambledRadicalInverseSpecialized<1811>(a, seed);
    case 280:
        return ScrambledRadicalInverseSpecialized<1823>(a, seed);
    case 281:
        return ScrambledRadicalInverseSpecialized<1831>(a, seed);
    case 282:
        return ScrambledRadicalInverseSpecialized<1847>(a, seed);
    case 283:
        return ScrambledRadicalInverseSpecialized<1861>(a, seed);
    case 284:
        return ScrambledRadicalInverseSpecialized<1867>(a, seed);
    case 285:
        return ScrambledRadicalInverseSpecialized<1871>(a, seed);
    case 286:
        return ScrambledRadicalInverseSpecialized<1873>(a, seed);
    case 287:
        return ScrambledRadicalInverseSpecialized<1877>(a, seed);
    case 288:
        return ScrambledRadicalInverseSpecialized<1879>(a, seed);
    case 289:
        return ScrambledRadicalInverseSpecialized<1889>(a, seed);
    case 290:
        return ScrambledRadicalInverseSpecialized<1901>(a, seed);
    case 291:
        return ScrambledRadicalInverseSpecialized<1907>(a, seed);
    case 292:
        return ScrambledRadicalInverseSpecialized<1913>(a, seed);
    case 293:
        return ScrambledRadicalInverseSpecialized<1931>(a, seed);
    case 294:
        return ScrambledRadicalInverseSpecialized<1933>(a, seed);
    case 295:
        return ScrambledRadicalInverseSpecialized<1949>(a, seed);
    case 296:
        return ScrambledRadicalInverseSpecialized<1951>(a, seed);
    case 297:
        return ScrambledRadicalInverseSpecialized<1973>(a, seed);
    case 298:
        return ScrambledRadicalInverseSpecialized<1979>(a, seed);
    case 299:
        return ScrambledRadicalInverseSpecialized<1987>(a, seed);
    case 300:
        return ScrambledRadicalInverseSpecialized<1993>(a, seed);
    case 301:
        return ScrambledRadicalInverseSpecialized<1997>(a, seed);
    case 302:
        return ScrambledRadicalInverseSpecialized<1999>(a, seed);
    case 303:
        return ScrambledRadicalInverseSpecialized<2003>(a, seed);
    case 304:
        return ScrambledRadicalInverseSpecialized<2011>(a, seed);
    case 305:
        return ScrambledRadicalInverseSpecialized<2017>(a, seed);
    case 306:
        return ScrambledRadicalInverseSpecialized<2027>(a, seed);
    case 307:
        return ScrambledRadicalInverseSpecialized<2029>(a, seed);
    case 308:
        return ScrambledRadicalInverseSpecialized<2039>(a, seed);
    case 309:
        return ScrambledRadicalInverseSpecialized<2053>(a, seed);
    case 310:
        return ScrambledRadicalInverseSpecialized<2063>(a, seed);
    case 311:
        return ScrambledRadicalInverseSpecialized<2069>(a, seed);
    case 312:
        return ScrambledRadicalInverseSpecialized<2081>(a, seed);
    case 313:
        return ScrambledRadicalInverseSpecialized<2083>(a, seed);
    case 314:
        return ScrambledRadicalInverseSpecialized<2087>(a, seed);
    case 315:
        return ScrambledRadicalInverseSpecialized<2089>(a, seed);
    case 316:
        return ScrambledRadicalInverseSpecialized<2099>(a, seed);
    case 317:
        return ScrambledRadicalInverseSpecialized<2111>(a, seed);
    case 318:
        return ScrambledRadicalInverseSpecialized<2113>(a, seed);
    case 319:
        return ScrambledRadicalInverseSpecialized<2129>(a, seed);
    case 320:
        return ScrambledRadicalInverseSpecialized<2131>(a, seed);
    case 321:
        return ScrambledRadicalInverseSpecialized<2137>(a, seed);
    case 322:
        return ScrambledRadicalInverseSpecialized<2141>(a, seed);
    case 323:
        return ScrambledRadicalInverseSpecialized<2143>(a, seed);
    case 324:
        return ScrambledRadicalInverseSpecialized<2153>(a, seed);
    case 325:
        return ScrambledRadicalInverseSpecialized<2161>(a, seed);
    case 326:
        return ScrambledRadicalInverseSpecialized<2179>(a, seed);
    case 327:
        return ScrambledRadicalInverseSpecialized<2203>(a, seed);
    case 328:
        return ScrambledRadicalInverseSpecialized<2207>(a, seed);
    case 329:
        return ScrambledRadicalInverseSpecialized<2213>(a, seed);
    case 330:
        return ScrambledRadicalInverseSpecialized<2221>(a, seed);
    case 331:
        return ScrambledRadicalInverseSpecialized<2237>(a, seed);
    case 332:
        return ScrambledRadicalInverseSpecialized<2239>(a, seed);
    case 333:
        return ScrambledRadicalInverseSpecialized<2243>(a, seed);
    case 334:
        return ScrambledRadicalInverseSpecialized<2251>(a, seed);
    case 335:
        return ScrambledRadicalInverseSpecialized<2267>(a, seed);
    case 336:
        return ScrambledRadicalInverseSpecialized<2269>(a, seed);
    case 337:
        return ScrambledRadicalInverseSpecialized<2273>(a, seed);
    case 338:
        return ScrambledRadicalInverseSpecialized<2281>(a, seed);
    case 339:
        return ScrambledRadicalInverseSpecialized<2287>(a, seed);
    case 340:
        return ScrambledRadicalInverseSpecialized<2293>(a, seed);
    case 341:
        return ScrambledRadicalInverseSpecialized<2297>(a, seed);
    case 342:
        return ScrambledRadicalInverseSpecialized<2309>(a, seed);
    case 343:
        return ScrambledRadicalInverseSpecialized<2311>(a, seed);
    case 344:
        return ScrambledRadicalInverseSpecialized<2333>(a, seed);
    case 345:
        return ScrambledRadicalInverseSpecialized<2339>(a, seed);
    case 346:
        return ScrambledRadicalInverseSpecialized<2341>(a, seed);
    case 347:
        return ScrambledRadicalInverseSpecialized<2347>(a, seed);
    case 348:
        return ScrambledRadicalInverseSpecialized<2351>(a, seed);
    case 349:
        return ScrambledRadicalInverseSpecialized<2357>(a, seed);
    case 350:
        return ScrambledRadicalInverseSpecialized<2371>(a, seed);
    case 351:
        return ScrambledRadicalInverseSpecialized<2377>(a, seed);
    case 352:
        return ScrambledRadicalInverseSpecialized<2381>(a, seed);
    case 353:
        return ScrambledRadicalInverseSpecialized<2383>(a, seed);
    case 354:
        return ScrambledRadicalInverseSpecialized<2389>(a, seed);
    case 355:
        return ScrambledRadicalInverseSpecialized<2393>(a, seed);
    case 356:
        return ScrambledRadicalInverseSpecialized<2399>(a, seed);
    case 357:
        return ScrambledRadicalInverseSpecialized<2411>(a, seed);
    case 358:
        return ScrambledRadicalInverseSpecialized<2417>(a, seed);
    case 359:
        return ScrambledRadicalInverseSpecialized<2423>(a, seed);
    case 360:
        return ScrambledRadicalInverseSpecialized<2437>(a, seed);
    case 361:
        return ScrambledRadicalInverseSpecialized<2441>(a, seed);
    case 362:
        return ScrambledRadicalInverseSpecialized<2447>(a, seed);
    case 363:
        return ScrambledRadicalInverseSpecialized<2459>(a, seed);
    case 364:
        return ScrambledRadicalInverseSpecialized<2467>(a, seed);
    case 365:
        return ScrambledRadicalInverseSpecialized<2473>(a, seed);
    case 366:
        return ScrambledRadicalInverseSpecialized<2477>(a, seed);
    case 367:
        return ScrambledRadicalInverseSpecialized<2503>(a, seed);
    case 368:
        return ScrambledRadicalInverseSpecialized<2521>(a, seed);
    case 369:
        return ScrambledRadicalInverseSpecialized<2531>(a, seed);
    case 370:
        return ScrambledRadicalInverseSpecialized<2539>(a, seed);
    case 371:
        return ScrambledRadicalInverseSpecialized<2543>(a, seed);
    case 372:
        return ScrambledRadicalInverseSpecialized<2549>(a, seed);
    case 373:
        return ScrambledRadicalInverseSpecialized<2551>(a, seed);
    case 374:
        return ScrambledRadicalInverseSpecialized<2557>(a, seed);
    case 375:
        return ScrambledRadicalInverseSpecialized<2579>(a, seed);
    case 376:
        return ScrambledRadicalInverseSpecialized<2591>(a, seed);
    case 377:
        return ScrambledRadicalInverseSpecialized<2593>(a, seed);
    case 378:
        return ScrambledRadicalInverseSpecialized<2609>(a, seed);
    case 379:
        return ScrambledRadicalInverseSpecialized<2617>(a, seed);
    case 380:
        return ScrambledRadicalInverseSpecialized<2621>(a, seed);
    case 381:
        return ScrambledRadicalInverseSpecialized<2633>(a, seed);
    case 382:
        return ScrambledRadicalInverseSpecialized<2647>(a, seed);
    case 383:
        return ScrambledRadicalInverseSpecialized<2657>(a, seed);
    case 384:
        return ScrambledRadicalInverseSpecialized<2659>(a, seed);
    case 385:
        return ScrambledRadicalInverseSpecialized<2663>(a, seed);
    case 386:
        return ScrambledRadicalInverseSpecialized<2671>(a, seed);
    case 387:
        return ScrambledRadicalInverseSpecialized<2677>(a, seed);
    case 388:
        return ScrambledRadicalInverseSpecialized<2683>(a, seed);
    case 389:
        return ScrambledRadicalInverseSpecialized<2687>(a, seed);
    case 390:
        return ScrambledRadicalInverseSpecialized<2689>(a, seed);
    case 391:
        return ScrambledRadicalInverseSpecialized<2693>(a, seed);
    case 392:
        return ScrambledRadicalInverseSpecialized<2699>(a, seed);
    case 393:
        return ScrambledRadicalInverseSpecialized<2707>(a, seed);
    case 394:
        return ScrambledRadicalInverseSpecialized<2711>(a, seed);
    case 395:
        return ScrambledRadicalInverseSpecialized<2713>(a, seed);
    case 396:
        return ScrambledRadicalInverseSpecialized<2719>(a, seed);
    case 397:
        return ScrambledRadicalInverseSpecialized<2729>(a, seed);
    case 398:
        return ScrambledRadicalInverseSpecialized<2731>(a, seed);
    case 399:
        return ScrambledRadicalInverseSpecialized<2741>(a, seed);
    case 400:
        return ScrambledRadicalInverseSpecialized<2749>(a, seed);
    case 401:
        return ScrambledRadicalInverseSpecialized<2753>(a, seed);
    case 402:
        return ScrambledRadicalInverseSpecialized<2767>(a, seed);
    case 403:
        return ScrambledRadicalInverseSpecialized<2777>(a, seed);
    case 404:
        return ScrambledRadicalInverseSpecialized<2789>(a, seed);
    case 405:
        return ScrambledRadicalInverseSpecialized<2791>(a, seed);
    case 406:
        return ScrambledRadicalInverseSpecialized<2797>(a, seed);
    case 407:
        return ScrambledRadicalInverseSpecialized<2801>(a, seed);
    case 408:
        return ScrambledRadicalInverseSpecialized<2803>(a, seed);
    case 409:
        return ScrambledRadicalInverseSpecialized<2819>(a, seed);
    case 410:
        return ScrambledRadicalInverseSpecialized<2833>(a, seed);
    case 411:
        return ScrambledRadicalInverseSpecialized<2837>(a, seed);
    case 412:
        return ScrambledRadicalInverseSpecialized<2843>(a, seed);
    case 413:
        return ScrambledRadicalInverseSpecialized<2851>(a, seed);
    case 414:
        return ScrambledRadicalInverseSpecialized<2857>(a, seed);
    case 415:
        return ScrambledRadicalInverseSpecialized<2861>(a, seed);
    case 416:
        return ScrambledRadicalInverseSpecialized<2879>(a, seed);
    case 417:
        return ScrambledRadicalInverseSpecialized<2887>(a, seed);
    case 418:
        return ScrambledRadicalInverseSpecialized<2897>(a, seed);
    case 419:
        return ScrambledRadicalInverseSpecialized<2903>(a, seed);
    case 420:
        return ScrambledRadicalInverseSpecialized<2909>(a, seed);
    case 421:
        return ScrambledRadicalInverseSpecialized<2917>(a, seed);
    case 422:
        return ScrambledRadicalInverseSpecialized<2927>(a, seed);
    case 423:
        return ScrambledRadicalInverseSpecialized<2939>(a, seed);
    case 424:
        return ScrambledRadicalInverseSpecialized<2953>(a, seed);
    case 425:
        return ScrambledRadicalInverseSpecialized<2957>(a, seed);
    case 426:
        return ScrambledRadicalInverseSpecialized<2963>(a, seed);
    case 427:
        return ScrambledRadicalInverseSpecialized<2969>(a, seed);
    case 428:
        return ScrambledRadicalInverseSpecialized<2971>(a, seed);
    case 429:
        return ScrambledRadicalInverseSpecialized<2999>(a, seed);
    case 430:
        return ScrambledRadicalInverseSpecialized<3001>(a, seed);
    case 431:
        return ScrambledRadicalInverseSpecialized<3011>(a, seed);
    case 432:
        return ScrambledRadicalInverseSpecialized<3019>(a, seed);
    case 433:
        return ScrambledRadicalInverseSpecialized<3023>(a, seed);
    case 434:
        return ScrambledRadicalInverseSpecialized<3037>(a, seed);
    case 435:
        return ScrambledRadicalInverseSpecialized<3041>(a, seed);
    case 436:
        return ScrambledRadicalInverseSpecialized<3049>(a, seed);
    case 437:
        return ScrambledRadicalInverseSpecialized<3061>(a, seed);
    case 438:
        return ScrambledRadicalInverseSpecialized<3067>(a, seed);
    case 439:
        return ScrambledRadicalInverseSpecialized<3079>(a, seed);
    case 440:
        return ScrambledRadicalInverseSpecialized<3083>(a, seed);
    case 441:
        return ScrambledRadicalInverseSpecialized<3089>(a, seed);
    case 442:
        return ScrambledRadicalInverseSpecialized<3109>(a, seed);
    case 443:
        return ScrambledRadicalInverseSpecialized<3119>(a, seed);
    case 444:
        return ScrambledRadicalInverseSpecialized<3121>(a, seed);
    case 445:
        return ScrambledRadicalInverseSpecialized<3137>(a, seed);
    case 446:
        return ScrambledRadicalInverseSpecialized<3163>(a, seed);
    case 447:
        return ScrambledRadicalInverseSpecialized<3167>(a, seed);
    case 448:
        return ScrambledRadicalInverseSpecialized<3169>(a, seed);
    case 449:
        return ScrambledRadicalInverseSpecialized<3181>(a, seed);
    case 450:
        return ScrambledRadicalInverseSpecialized<3187>(a, seed);
    case 451:
        return ScrambledRadicalInverseSpecialized<3191>(a, seed);
    case 452:
        return ScrambledRadicalInverseSpecialized<3203>(a, seed);
    case 453:
        return ScrambledRadicalInverseSpecialized<3209>(a, seed);
    case 454:
        return ScrambledRadicalInverseSpecialized<3217>(a, seed);
    case 455:
        return ScrambledRadicalInverseSpecialized<3221>(a, seed);
    case 456:
        return ScrambledRadicalInverseSpecialized<3229>(a, seed);
    case 457:
        return ScrambledRadicalInverseSpecialized<3251>(a, seed);
    case 458:
        return ScrambledRadicalInverseSpecialized<3253>(a, seed);
    case 459:
        return ScrambledRadicalInverseSpecialized<3257>(a, seed);
    case 460:
        return ScrambledRadicalInverseSpecialized<3259>(a, seed);
    case 461:
        return ScrambledRadicalInverseSpecialized<3271>(a, seed);
    case 462:
        return ScrambledRadicalInverseSpecialized<3299>(a, seed);
    case 463:
        return ScrambledRadicalInverseSpecialized<3301>(a, seed);
    case 464:
        return ScrambledRadicalInverseSpecialized<3307>(a, seed);
    case 465:
        return ScrambledRadicalInverseSpecialized<3313>(a, seed);
    case 466:
        return ScrambledRadicalInverseSpecialized<3319>(a, seed);
    case 467:
        return ScrambledRadicalInverseSpecialized<3323>(a, seed);
    case 468:
        return ScrambledRadicalInverseSpecialized<3329>(a, seed);
    case 469:
        return ScrambledRadicalInverseSpecialized<3331>(a, seed);
    case 470:
        return ScrambledRadicalInverseSpecialized<3343>(a, seed);
    case 471:
        return ScrambledRadicalInverseSpecialized<3347>(a, seed);
    case 472:
        return ScrambledRadicalInverseSpecialized<3359>(a, seed);
    case 473:
        return ScrambledRadicalInverseSpecialized<3361>(a, seed);
    case 474:
        return ScrambledRadicalInverseSpecialized<3371>(a, seed);
    case 475:
        return ScrambledRadicalInverseSpecialized<3373>(a, seed);
    case 476:
        return ScrambledRadicalInverseSpecialized<3389>(a, seed);
    case 477:
        return ScrambledRadicalInverseSpecialized<3391>(a, seed);
    case 478:
        return ScrambledRadicalInverseSpecialized<3407>(a, seed);
    case 479:
        return ScrambledRadicalInverseSpecialized<3413>(a, seed);
    case 480:
        return ScrambledRadicalInverseSpecialized<3433>(a, seed);
    case 481:
        return ScrambledRadicalInverseSpecialized<3449>(a, seed);
    case 482:
        return ScrambledRadicalInverseSpecialized<3457>(a, seed);
    case 483:
        return ScrambledRadicalInverseSpecialized<3461>(a, seed);
    case 484:
        return ScrambledRadicalInverseSpecialized<3463>(a, seed);
    case 485:
        return ScrambledRadicalInverseSpecialized<3467>(a, seed);
    case 486:
        return ScrambledRadicalInverseSpecialized<3469>(a, seed);
    case 487:
        return ScrambledRadicalInverseSpecialized<3491>(a, seed);
    case 488:
        return ScrambledRadicalInverseSpecialized<3499>(a, seed);
    case 489:
        return ScrambledRadicalInverseSpecialized<3511>(a, seed);
    case 490:
        return ScrambledRadicalInverseSpecialized<3517>(a, seed);
    case 491:
        return ScrambledRadicalInverseSpecialized<3527>(a, seed);
    case 492:
        return ScrambledRadicalInverseSpecialized<3529>(a, seed);
    case 493:
        return ScrambledRadicalInverseSpecialized<3533>(a, seed);
    case 494:
        return ScrambledRadicalInverseSpecialized<3539>(a, seed);
    case 495:
        return ScrambledRadicalInverseSpecialized<3541>(a, seed);
    case 496:
        return ScrambledRadicalInverseSpecialized<3547>(a, seed);
    case 497:
        return ScrambledRadicalInverseSpecialized<3557>(a, seed);
    case 498:
        return ScrambledRadicalInverseSpecialized<3559>(a, seed);
    case 499:
        return ScrambledRadicalInverseSpecialized<3571>(a, seed);
    case 500:
        return ScrambledRadicalInverseSpecialized<3581>(a, seed);
    case 501:
        return ScrambledRadicalInverseSpecialized<3583>(a, seed);
    case 502:
        return ScrambledRadicalInverseSpecialized<3593>(a, seed);
    case 503:
        return ScrambledRadicalInverseSpecialized<3607>(a, seed);
    case 504:
        return ScrambledRadicalInverseSpecialized<3613>(a, seed);
    case 505:
        return ScrambledRadicalInverseSpecialized<3617>(a, seed);
    case 506:
        return ScrambledRadicalInverseSpecialized<3623>(a, seed);
    case 507:
        return ScrambledRadicalInverseSpecialized<3631>(a, seed);
    case 508:
        return ScrambledRadicalInverseSpecialized<3637>(a, seed);
    case 509:
        return ScrambledRadicalInverseSpecialized<3643>(a, seed);
    case 510:
        return ScrambledRadicalInverseSpecialized<3659>(a, seed);
    case 511:
        return ScrambledRadicalInverseSpecialized<3671>(a, seed);
    case 512:
        return ScrambledRadicalInverseSpecialized<3673>(a, seed);
    case 513:
        return ScrambledRadicalInverseSpecialized<3677>(a, seed);
    case 514:
        return ScrambledRadicalInverseSpecialized<3691>(a, seed);
    case 515:
        return ScrambledRadicalInverseSpecialized<3697>(a, seed);
    case 516:
        return ScrambledRadicalInverseSpecialized<3701>(a, seed);
    case 517:
        return ScrambledRadicalInverseSpecialized<3709>(a, seed);
    case 518:
        return ScrambledRadicalInverseSpecialized<3719>(a, seed);
    case 519:
        return ScrambledRadicalInverseSpecialized<3727>(a, seed);
    case 520:
        return ScrambledRadicalInverseSpecialized<3733>(a, seed);
    case 521:
        return ScrambledRadicalInverseSpecialized<3739>(a, seed);
    case 522:
        return ScrambledRadicalInverseSpecialized<3761>(a, seed);
    case 523:
        return ScrambledRadicalInverseSpecialized<3767>(a, seed);
    case 524:
        return ScrambledRadicalInverseSpecialized<3769>(a, seed);
    case 525:
        return ScrambledRadicalInverseSpecialized<3779>(a, seed);
    case 526:
        return ScrambledRadicalInverseSpecialized<3793>(a, seed);
    case 527:
        return ScrambledRadicalInverseSpecialized<3797>(a, seed);
    case 528:
        return ScrambledRadicalInverseSpecialized<3803>(a, seed);
    case 529:
        return ScrambledRadicalInverseSpecialized<3821>(a, seed);
    case 530:
        return ScrambledRadicalInverseSpecialized<3823>(a, seed);
    case 531:
        return ScrambledRadicalInverseSpecialized<3833>(a, seed);
    case 532:
        return ScrambledRadicalInverseSpecialized<3847>(a, seed);
    case 533:
        return ScrambledRadicalInverseSpecialized<3851>(a, seed);
    case 534:
        return ScrambledRadicalInverseSpecialized<3853>(a, seed);
    case 535:
        return ScrambledRadicalInverseSpecialized<3863>(a, seed);
    case 536:
        return ScrambledRadicalInverseSpecialized<3877>(a, seed);
    case 537:
        return ScrambledRadicalInverseSpecialized<3881>(a, seed);
    case 538:
        return ScrambledRadicalInverseSpecialized<3889>(a, seed);
    case 539:
        return ScrambledRadicalInverseSpecialized<3907>(a, seed);
    case 540:
        return ScrambledRadicalInverseSpecialized<3911>(a, seed);
    case 541:
        return ScrambledRadicalInverseSpecialized<3917>(a, seed);
    case 542:
        return ScrambledRadicalInverseSpecialized<3919>(a, seed);
    case 543:
        return ScrambledRadicalInverseSpecialized<3923>(a, seed);
    case 544:
        return ScrambledRadicalInverseSpecialized<3929>(a, seed);
    case 545:
        return ScrambledRadicalInverseSpecialized<3931>(a, seed);
    case 546:
        return ScrambledRadicalInverseSpecialized<3943>(a, seed);
    case 547:
        return ScrambledRadicalInverseSpecialized<3947>(a, seed);
    case 548:
        return ScrambledRadicalInverseSpecialized<3967>(a, seed);
    case 549:
        return ScrambledRadicalInverseSpecialized<3989>(a, seed);
    case 550:
        return ScrambledRadicalInverseSpecialized<4001>(a, seed);
    case 551:
        return ScrambledRadicalInverseSpecialized<4003>(a, seed);
    case 552:
        return ScrambledRadicalInverseSpecialized<4007>(a, seed);
    case 553:
        return ScrambledRadicalInverseSpecialized<4013>(a, seed);
    case 554:
        return ScrambledRadicalInverseSpecialized<4019>(a, seed);
    case 555:
        return ScrambledRadicalInverseSpecialized<4021>(a, seed);
    case 556:
        return ScrambledRadicalInverseSpecialized<4027>(a, seed);
    case 557:
        return ScrambledRadicalInverseSpecialized<4049>(a, seed);
    case 558:
        return ScrambledRadicalInverseSpecialized<4051>(a, seed);
    case 559:
        return ScrambledRadicalInverseSpecialized<4057>(a, seed);
    case 560:
        return ScrambledRadicalInverseSpecialized<4073>(a, seed);
    case 561:
        return ScrambledRadicalInverseSpecialized<4079>(a, seed);
    case 562:
        return ScrambledRadicalInverseSpecialized<4091>(a, seed);
    case 563:
        return ScrambledRadicalInverseSpecialized<4093>(a, seed);
    case 564:
        return ScrambledRadicalInverseSpecialized<4099>(a, seed);
    case 565:
        return ScrambledRadicalInverseSpecialized<4111>(a, seed);
    case 566:
        return ScrambledRadicalInverseSpecialized<4127>(a, seed);
    case 567:
        return ScrambledRadicalInverseSpecialized<4129>(a, seed);
    case 568:
        return ScrambledRadicalInverseSpecialized<4133>(a, seed);
    case 569:
        return ScrambledRadicalInverseSpecialized<4139>(a, seed);
    case 570:
        return ScrambledRadicalInverseSpecialized<4153>(a, seed);
    case 571:
        return ScrambledRadicalInverseSpecialized<4157>(a, seed);
    case 572:
        return ScrambledRadicalInverseSpecialized<4159>(a, seed);
    case 573:
        return ScrambledRadicalInverseSpecialized<4177>(a, seed);
    case 574:
        return ScrambledRadicalInverseSpecialized<4201>(a, seed);
    case 575:
        return ScrambledRadicalInverseSpecialized<4211>(a, seed);
    case 576:
        return ScrambledRadicalInverseSpecialized<4217>(a, seed);
    case 577:
        return ScrambledRadicalInverseSpecialized<4219>(a, seed);
    case 578:
        return ScrambledRadicalInverseSpecialized<4229>(a, seed);
    case 579:
        return ScrambledRadicalInverseSpecialized<4231>(a, seed);
    case 580:
        return ScrambledRadicalInverseSpecialized<4241>(a, seed);
    case 581:
        return ScrambledRadicalInverseSpecialized<4243>(a, seed);
    case 582:
        return ScrambledRadicalInverseSpecialized<4253>(a, seed);
    case 583:
        return ScrambledRadicalInverseSpecialized<4259>(a, seed);
    case 584:
        return ScrambledRadicalInverseSpecialized<4261>(a, seed);
    case 585:
        return ScrambledRadicalInverseSpecialized<4271>(a, seed);
    case 586:
        return ScrambledRadicalInverseSpecialized<4273>(a, seed);
    case 587:
        return ScrambledRadicalInverseSpecialized<4283>(a, seed);
    case 588:
        return ScrambledRadicalInverseSpecialized<4289>(a, seed);
    case 589:
        return ScrambledRadicalInverseSpecialized<4297>(a, seed);
    case 590:
        return ScrambledRadicalInverseSpecialized<4327>(a, seed);
    case 591:
        return ScrambledRadicalInverseSpecialized<4337>(a, seed);
    case 592:
        return ScrambledRadicalInverseSpecialized<4339>(a, seed);
    case 593:
        return ScrambledRadicalInverseSpecialized<4349>(a, seed);
    case 594:
        return ScrambledRadicalInverseSpecialized<4357>(a, seed);
    case 595:
        return ScrambledRadicalInverseSpecialized<4363>(a, seed);
    case 596:
        return ScrambledRadicalInverseSpecialized<4373>(a, seed);
    case 597:
        return ScrambledRadicalInverseSpecialized<4391>(a, seed);
    case 598:
        return ScrambledRadicalInverseSpecialized<4397>(a, seed);
    case 599:
        return ScrambledRadicalInverseSpecialized<4409>(a, seed);
    case 600:
        return ScrambledRadicalInverseSpecialized<4421>(a, seed);
    case 601:
        return ScrambledRadicalInverseSpecialized<4423>(a, seed);
    case 602:
        return ScrambledRadicalInverseSpecialized<4441>(a, seed);
    case 603:
        return ScrambledRadicalInverseSpecialized<4447>(a, seed);
    case 604:
        return ScrambledRadicalInverseSpecialized<4451>(a, seed);
    case 605:
        return ScrambledRadicalInverseSpecialized<4457>(a, seed);
    case 606:
        return ScrambledRadicalInverseSpecialized<4463>(a, seed);
    case 607:
        return ScrambledRadicalInverseSpecialized<4481>(a, seed);
    case 608:
        return ScrambledRadicalInverseSpecialized<4483>(a, seed);
    case 609:
        return ScrambledRadicalInverseSpecialized<4493>(a, seed);
    case 610:
        return ScrambledRadicalInverseSpecialized<4507>(a, seed);
    case 611:
        return ScrambledRadicalInverseSpecialized<4513>(a, seed);
    case 612:
        return ScrambledRadicalInverseSpecialized<4517>(a, seed);
    case 613:
        return ScrambledRadicalInverseSpecialized<4519>(a, seed);
    case 614:
        return ScrambledRadicalInverseSpecialized<4523>(a, seed);
    case 615:
        return ScrambledRadicalInverseSpecialized<4547>(a, seed);
    case 616:
        return ScrambledRadicalInverseSpecialized<4549>(a, seed);
    case 617:
        return ScrambledRadicalInverseSpecialized<4561>(a, seed);
    case 618:
        return ScrambledRadicalInverseSpecialized<4567>(a, seed);
    case 619:
        return ScrambledRadicalInverseSpecialized<4583>(a, seed);
    case 620:
        return ScrambledRadicalInverseSpecialized<4591>(a, seed);
    case 621:
        return ScrambledRadicalInverseSpecialized<4597>(a, seed);
    case 622:
        return ScrambledRadicalInverseSpecialized<4603>(a, seed);
    case 623:
        return ScrambledRadicalInverseSpecialized<4621>(a, seed);
    case 624:
        return ScrambledRadicalInverseSpecialized<4637>(a, seed);
    case 625:
        return ScrambledRadicalInverseSpecialized<4639>(a, seed);
    case 626:
        return ScrambledRadicalInverseSpecialized<4643>(a, seed);
    case 627:
        return ScrambledRadicalInverseSpecialized<4649>(a, seed);
    case 628:
        return ScrambledRadicalInverseSpecialized<4651>(a, seed);
    case 629:
        return ScrambledRadicalInverseSpecialized<4657>(a, seed);
    case 630:
        return ScrambledRadicalInverseSpecialized<4663>(a, seed);
    case 631:
        return ScrambledRadicalInverseSpecialized<4673>(a, seed);
    case 632:
        return ScrambledRadicalInverseSpecialized<4679>(a, seed);
    case 633:
        return ScrambledRadicalInverseSpecialized<4691>(a, seed);
    case 634:
        return ScrambledRadicalInverseSpecialized<4703>(a, seed);
    case 635:
        return ScrambledRadicalInverseSpecialized<4721>(a, seed);
    case 636:
        return ScrambledRadicalInverseSpecialized<4723>(a, seed);
    case 637:
        return ScrambledRadicalInverseSpecialized<4729>(a, seed);
    case 638:
        return ScrambledRadicalInverseSpecialized<4733>(a, seed);
    case 639:
        return ScrambledRadicalInverseSpecialized<4751>(a, seed);
    case 640:
        return ScrambledRadicalInverseSpecialized<4759>(a, seed);
    case 641:
        return ScrambledRadicalInverseSpecialized<4783>(a, seed);
    case 642:
        return ScrambledRadicalInverseSpecialized<4787>(a, seed);
    case 643:
        return ScrambledRadicalInverseSpecialized<4789>(a, seed);
    case 644:
        return ScrambledRadicalInverseSpecialized<4793>(a, seed);
    case 645:
        return ScrambledRadicalInverseSpecialized<4799>(a, seed);
    case 646:
        return ScrambledRadicalInverseSpecialized<4801>(a, seed);
    case 647:
        return ScrambledRadicalInverseSpecialized<4813>(a, seed);
    case 648:
        return ScrambledRadicalInverseSpecialized<4817>(a, seed);
    case 649:
        return ScrambledRadicalInverseSpecialized<4831>(a, seed);
    case 650:
        return ScrambledRadicalInverseSpecialized<4861>(a, seed);
    case 651:
        return ScrambledRadicalInverseSpecialized<4871>(a, seed);
    case 652:
        return ScrambledRadicalInverseSpecialized<4877>(a, seed);
    case 653:
        return ScrambledRadicalInverseSpecialized<4889>(a, seed);
    case 654:
        return ScrambledRadicalInverseSpecialized<4903>(a, seed);
    case 655:
        return ScrambledRadicalInverseSpecialized<4909>(a, seed);
    case 656:
        return ScrambledRadicalInverseSpecialized<4919>(a, seed);
    case 657:
        return ScrambledRadicalInverseSpecialized<4931>(a, seed);
    case 658:
        return ScrambledRadicalInverseSpecialized<4933>(a, seed);
    case 659:
        return ScrambledRadicalInverseSpecialized<4937>(a, seed);
    case 660:
        return ScrambledRadicalInverseSpecialized<4943>(a, seed);
    case 661:
        return ScrambledRadicalInverseSpecialized<4951>(a, seed);
    case 662:
        return ScrambledRadicalInverseSpecialized<4957>(a, seed);
    case 663:
        return ScrambledRadicalInverseSpecialized<4967>(a, seed);
    case 664:
        return ScrambledRadicalInverseSpecialized<4969>(a, seed);
    case 665:
        return ScrambledRadicalInverseSpecialized<4973>(a, seed);
    case 666:
        return ScrambledRadicalInverseSpecialized<4987>(a, seed);
    case 667:
        return ScrambledRadicalInverseSpecialized<4993>(a, seed);
    case 668:
        return ScrambledRadicalInverseSpecialized<4999>(a, seed);
    case 669:
        return ScrambledRadicalInverseSpecialized<5003>(a, seed);
    case 670:
        return ScrambledRadicalInverseSpecialized<5009>(a, seed);
    case 671:
        return ScrambledRadicalInverseSpecialized<5011>(a, seed);
    case 672:
        return ScrambledRadicalInverseSpecialized<5021>(a, seed);
    case 673:
        return ScrambledRadicalInverseSpecialized<5023>(a, seed);
    case 674:
        return ScrambledRadicalInverseSpecialized<5039>(a, seed);
    case 675:
        return ScrambledRadicalInverseSpecialized<5051>(a, seed);
    case 676:
        return ScrambledRadicalInverseSpecialized<5059>(a, seed);
    case 677:
        return ScrambledRadicalInverseSpecialized<5077>(a, seed);
    case 678:
        return ScrambledRadicalInverseSpecialized<5081>(a, seed);
    case 679:
        return ScrambledRadicalInverseSpecialized<5087>(a, seed);
    case 680:
        return ScrambledRadicalInverseSpecialized<5099>(a, seed);
    case 681:
        return ScrambledRadicalInverseSpecialized<5101>(a, seed);
    case 682:
        return ScrambledRadicalInverseSpecialized<5107>(a, seed);
    case 683:
        return ScrambledRadicalInverseSpecialized<5113>(a, seed);
    case 684:
        return ScrambledRadicalInverseSpecialized<5119>(a, seed);
    case 685:
        return ScrambledRadicalInverseSpecialized<5147>(a, seed);
    case 686:
        return ScrambledRadicalInverseSpecialized<5153>(a, seed);
    case 687:
        return ScrambledRadicalInverseSpecialized<5167>(a, seed);
    case 688:
        return ScrambledRadicalInverseSpecialized<5171>(a, seed);
    case 689:
        return ScrambledRadicalInverseSpecialized<5179>(a, seed);
    case 690:
        return ScrambledRadicalInverseSpecialized<5189>(a, seed);
    case 691:
        return ScrambledRadicalInverseSpecialized<5197>(a, seed);
    case 692:
        return ScrambledRadicalInverseSpecialized<5209>(a, seed);
    case 693:
        return ScrambledRadicalInverseSpecialized<5227>(a, seed);
    case 694:
        return ScrambledRadicalInverseSpecialized<5231>(a, seed);
    case 695:
        return ScrambledRadicalInverseSpecialized<5233>(a, seed);
    case 696:
        return ScrambledRadicalInverseSpecialized<5237>(a, seed);
    case 697:
        return ScrambledRadicalInverseSpecialized<5261>(a, seed);
    case 698:
        return ScrambledRadicalInverseSpecialized<5273>(a, seed);
    case 699:
        return ScrambledRadicalInverseSpecialized<5279>(a, seed);
    case 700:
        return ScrambledRadicalInverseSpecialized<5281>(a, seed);
    case 701:
        return ScrambledRadicalInverseSpecialized<5297>(a, seed);
    case 702:
        return ScrambledRadicalInverseSpecialized<5303>(a, seed);
    case 703:
        return ScrambledRadicalInverseSpecialized<5309>(a, seed);
    case 704:
        return ScrambledRadicalInverseSpecialized<5323>(a, seed);
    case 705:
        return ScrambledRadicalInverseSpecialized<5333>(a, seed);
    case 706:
        return ScrambledRadicalInverseSpecialized<5347>(a, seed);
    case 707:
        return ScrambledRadicalInverseSpecialized<5351>(a, seed);
    case 708:
        return ScrambledRadicalInverseSpecialized<5381>(a, seed);
    case 709:
        return ScrambledRadicalInverseSpecialized<5387>(a, seed);
    case 710:
        return ScrambledRadicalInverseSpecialized<5393>(a, seed);
    case 711:
        return ScrambledRadicalInverseSpecialized<5399>(a, seed);
    case 712:
        return ScrambledRadicalInverseSpecialized<5407>(a, seed);
    case 713:
        return ScrambledRadicalInverseSpecialized<5413>(a, seed);
    case 714:
        return ScrambledRadicalInverseSpecialized<5417>(a, seed);
    case 715:
        return ScrambledRadicalInverseSpecialized<5419>(a, seed);
    case 716:
        return ScrambledRadicalInverseSpecialized<5431>(a, seed);
    case 717:
        return ScrambledRadicalInverseSpecialized<5437>(a, seed);
    case 718:
        return ScrambledRadicalInverseSpecialized<5441>(a, seed);
    case 719:
        return ScrambledRadicalInverseSpecialized<5443>(a, seed);
    case 720:
        return ScrambledRadicalInverseSpecialized<5449>(a, seed);
    case 721:
        return ScrambledRadicalInverseSpecialized<5471>(a, seed);
    case 722:
        return ScrambledRadicalInverseSpecialized<5477>(a, seed);
    case 723:
        return ScrambledRadicalInverseSpecialized<5479>(a, seed);
    case 724:
        return ScrambledRadicalInverseSpecialized<5483>(a, seed);
    case 725:
        return ScrambledRadicalInverseSpecialized<5501>(a, seed);
    case 726:
        return ScrambledRadicalInverseSpecialized<5503>(a, seed);
    case 727:
        return ScrambledRadicalInverseSpecialized<5507>(a, seed);
    case 728:
        return ScrambledRadicalInverseSpecialized<5519>(a, seed);
    case 729:
        return ScrambledRadicalInverseSpecialized<5521>(a, seed);
    case 730:
        return ScrambledRadicalInverseSpecialized<5527>(a, seed);
    case 731:
        return ScrambledRadicalInverseSpecialized<5531>(a, seed);
    case 732:
        return ScrambledRadicalInverseSpecialized<5557>(a, seed);
    case 733:
        return ScrambledRadicalInverseSpecialized<5563>(a, seed);
    case 734:
        return ScrambledRadicalInverseSpecialized<5569>(a, seed);
    case 735:
        return ScrambledRadicalInverseSpecialized<5573>(a, seed);
    case 736:
        return ScrambledRadicalInverseSpecialized<5581>(a, seed);
    case 737:
        return ScrambledRadicalInverseSpecialized<5591>(a, seed);
    case 738:
        return ScrambledRadicalInverseSpecialized<5623>(a, seed);
    case 739:
        return ScrambledRadicalInverseSpecialized<5639>(a, seed);
    case 740:
        return ScrambledRadicalInverseSpecialized<5641>(a, seed);
    case 741:
        return ScrambledRadicalInverseSpecialized<5647>(a, seed);
    case 742:
        return ScrambledRadicalInverseSpecialized<5651>(a, seed);
    case 743:
        return ScrambledRadicalInverseSpecialized<5653>(a, seed);
    case 744:
        return ScrambledRadicalInverseSpecialized<5657>(a, seed);
    case 745:
        return ScrambledRadicalInverseSpecialized<5659>(a, seed);
    case 746:
        return ScrambledRadicalInverseSpecialized<5669>(a, seed);
    case 747:
        return ScrambledRadicalInverseSpecialized<5683>(a, seed);
    case 748:
        return ScrambledRadicalInverseSpecialized<5689>(a, seed);
    case 749:
        return ScrambledRadicalInverseSpecialized<5693>(a, seed);
    case 750:
        return ScrambledRadicalInverseSpecialized<5701>(a, seed);
    case 751:
        return ScrambledRadicalInverseSpecialized<5711>(a, seed);
    case 752:
        return ScrambledRadicalInverseSpecialized<5717>(a, seed);
    case 753:
        return ScrambledRadicalInverseSpecialized<5737>(a, seed);
    case 754:
        return ScrambledRadicalInverseSpecialized<5741>(a, seed);
    case 755:
        return ScrambledRadicalInverseSpecialized<5743>(a, seed);
    case 756:
        return ScrambledRadicalInverseSpecialized<5749>(a, seed);
    case 757:
        return ScrambledRadicalInverseSpecialized<5779>(a, seed);
    case 758:
        return ScrambledRadicalInverseSpecialized<5783>(a, seed);
    case 759:
        return ScrambledRadicalInverseSpecialized<5791>(a, seed);
    case 760:
        return ScrambledRadicalInverseSpecialized<5801>(a, seed);
    case 761:
        return ScrambledRadicalInverseSpecialized<5807>(a, seed);
    case 762:
        return ScrambledRadicalInverseSpecialized<5813>(a, seed);
    case 763:
        return ScrambledRadicalInverseSpecialized<5821>(a, seed);
    case 764:
        return ScrambledRadicalInverseSpecialized<5827>(a, seed);
    case 765:
        return ScrambledRadicalInverseSpecialized<5839>(a, seed);
    case 766:
        return ScrambledRadicalInverseSpecialized<5843>(a, seed);
    case 767:
        return ScrambledRadicalInverseSpecialized<5849>(a, seed);
    case 768:
        return ScrambledRadicalInverseSpecialized<5851>(a, seed);
    case 769:
        return ScrambledRadicalInverseSpecialized<5857>(a, seed);
    case 770:
        return ScrambledRadicalInverseSpecialized<5861>(a, seed);
    case 771:
        return ScrambledRadicalInverseSpecialized<5867>(a, seed);
    case 772:
        return ScrambledRadicalInverseSpecialized<5869>(a, seed);
    case 773:
        return ScrambledRadicalInverseSpecialized<5879>(a, seed);
    case 774:
        return ScrambledRadicalInverseSpecialized<5881>(a, seed);
    case 775:
        return ScrambledRadicalInverseSpecialized<5897>(a, seed);
    case 776:
        return ScrambledRadicalInverseSpecialized<5903>(a, seed);
    case 777:
        return ScrambledRadicalInverseSpecialized<5923>(a, seed);
    case 778:
        return ScrambledRadicalInverseSpecialized<5927>(a, seed);
    case 779:
        return ScrambledRadicalInverseSpecialized<5939>(a, seed);
    case 780:
        return ScrambledRadicalInverseSpecialized<5953>(a, seed);
    case 781:
        return ScrambledRadicalInverseSpecialized<5981>(a, seed);
    case 782:
        return ScrambledRadicalInverseSpecialized<5987>(a, seed);
    case 783:
        return ScrambledRadicalInverseSpecialized<6007>(a, seed);
    case 784:
        return ScrambledRadicalInverseSpecialized<6011>(a, seed);
    case 785:
        return ScrambledRadicalInverseSpecialized<6029>(a, seed);
    case 786:
        return ScrambledRadicalInverseSpecialized<6037>(a, seed);
    case 787:
        return ScrambledRadicalInverseSpecialized<6043>(a, seed);
    case 788:
        return ScrambledRadicalInverseSpecialized<6047>(a, seed);
    case 789:
        return ScrambledRadicalInverseSpecialized<6053>(a, seed);
    case 790:
        return ScrambledRadicalInverseSpecialized<6067>(a, seed);
    case 791:
        return ScrambledRadicalInverseSpecialized<6073>(a, seed);
    case 792:
        return ScrambledRadicalInverseSpecialized<6079>(a, seed);
    case 793:
        return ScrambledRadicalInverseSpecialized<6089>(a, seed);
    case 794:
        return ScrambledRadicalInverseSpecialized<6091>(a, seed);
    case 795:
        return ScrambledRadicalInverseSpecialized<6101>(a, seed);
    case 796:
        return ScrambledRadicalInverseSpecialized<6113>(a, seed);
    case 797:
        return ScrambledRadicalInverseSpecialized<6121>(a, seed);
    case 798:
        return ScrambledRadicalInverseSpecialized<6131>(a, seed);
    case 799:
        return ScrambledRadicalInverseSpecialized<6133>(a, seed);
    case 800:
        return ScrambledRadicalInverseSpecialized<6143>(a, seed);
    case 801:
        return ScrambledRadicalInverseSpecialized<6151>(a, seed);
    case 802:
        return ScrambledRadicalInverseSpecialized<6163>(a, seed);
    case 803:
        return ScrambledRadicalInverseSpecialized<6173>(a, seed);
    case 804:
        return ScrambledRadicalInverseSpecialized<6197>(a, seed);
    case 805:
        return ScrambledRadicalInverseSpecialized<6199>(a, seed);
    case 806:
        return ScrambledRadicalInverseSpecialized<6203>(a, seed);
    case 807:
        return ScrambledRadicalInverseSpecialized<6211>(a, seed);
    case 808:
        return ScrambledRadicalInverseSpecialized<6217>(a, seed);
    case 809:
        return ScrambledRadicalInverseSpecialized<6221>(a, seed);
    case 810:
        return ScrambledRadicalInverseSpecialized<6229>(a, seed);
    case 811:
        return ScrambledRadicalInverseSpecialized<6247>(a, seed);
    case 812:
        return ScrambledRadicalInverseSpecialized<6257>(a, seed);
    case 813:
        return ScrambledRadicalInverseSpecialized<6263>(a, seed);
    case 814:
        return ScrambledRadicalInverseSpecialized<6269>(a, seed);
    case 815:
        return ScrambledRadicalInverseSpecialized<6271>(a, seed);
    case 816:
        return ScrambledRadicalInverseSpecialized<6277>(a, seed);
    case 817:
        return ScrambledRadicalInverseSpecialized<6287>(a, seed);
    case 818:
        return ScrambledRadicalInverseSpecialized<6299>(a, seed);
    case 819:
        return ScrambledRadicalInverseSpecialized<6301>(a, seed);
    case 820:
        return ScrambledRadicalInverseSpecialized<6311>(a, seed);
    case 821:
        return ScrambledRadicalInverseSpecialized<6317>(a, seed);
    case 822:
        return ScrambledRadicalInverseSpecialized<6323>(a, seed);
    case 823:
        return ScrambledRadicalInverseSpecialized<6329>(a, seed);
    case 824:
        return ScrambledRadicalInverseSpecialized<6337>(a, seed);
    case 825:
        return ScrambledRadicalInverseSpecialized<6343>(a, seed);
    case 826:
        return ScrambledRadicalInverseSpecialized<6353>(a, seed);
    case 827:
        return ScrambledRadicalInverseSpecialized<6359>(a, seed);
    case 828:
        return ScrambledRadicalInverseSpecialized<6361>(a, seed);
    case 829:
        return ScrambledRadicalInverseSpecialized<6367>(a, seed);
    case 830:
        return ScrambledRadicalInverseSpecialized<6373>(a, seed);
    case 831:
        return ScrambledRadicalInverseSpecialized<6379>(a, seed);
    case 832:
        return ScrambledRadicalInverseSpecialized<6389>(a, seed);
    case 833:
        return ScrambledRadicalInverseSpecialized<6397>(a, seed);
    case 834:
        return ScrambledRadicalInverseSpecialized<6421>(a, seed);
    case 835:
        return ScrambledRadicalInverseSpecialized<6427>(a, seed);
    case 836:
        return ScrambledRadicalInverseSpecialized<6449>(a, seed);
    case 837:
        return ScrambledRadicalInverseSpecialized<6451>(a, seed);
    case 838:
        return ScrambledRadicalInverseSpecialized<6469>(a, seed);
    case 839:
        return ScrambledRadicalInverseSpecialized<6473>(a, seed);
    case 840:
        return ScrambledRadicalInverseSpecialized<6481>(a, seed);
    case 841:
        return ScrambledRadicalInverseSpecialized<6491>(a, seed);
    case 842:
        return ScrambledRadicalInverseSpecialized<6521>(a, seed);
    case 843:
        return ScrambledRadicalInverseSpecialized<6529>(a, seed);
    case 844:
        return ScrambledRadicalInverseSpecialized<6547>(a, seed);
    case 845:
        return ScrambledRadicalInverseSpecialized<6551>(a, seed);
    case 846:
        return ScrambledRadicalInverseSpecialized<6553>(a, seed);
    case 847:
        return ScrambledRadicalInverseSpecialized<6563>(a, seed);
    case 848:
        return ScrambledRadicalInverseSpecialized<6569>(a, seed);
    case 849:
        return ScrambledRadicalInverseSpecialized<6571>(a, seed);
    case 850:
        return ScrambledRadicalInverseSpecialized<6577>(a, seed);
    case 851:
        return ScrambledRadicalInverseSpecialized<6581>(a, seed);
    case 852:
        return ScrambledRadicalInverseSpecialized<6599>(a, seed);
    case 853:
        return ScrambledRadicalInverseSpecialized<6607>(a, seed);
    case 854:
        return ScrambledRadicalInverseSpecialized<6619>(a, seed);
    case 855:
        return ScrambledRadicalInverseSpecialized<6637>(a, seed);
    case 856:
        return ScrambledRadicalInverseSpecialized<6653>(a, seed);
    case 857:
        return ScrambledRadicalInverseSpecialized<6659>(a, seed);
    case 858:
        return ScrambledRadicalInverseSpecialized<6661>(a, seed);
    case 859:
        return ScrambledRadicalInverseSpecialized<6673>(a, seed);
    case 860:
        return ScrambledRadicalInverseSpecialized<6679>(a, seed);
    case 861:
        return ScrambledRadicalInverseSpecialized<6689>(a, seed);
    case 862:
        return ScrambledRadicalInverseSpecialized<6691>(a, seed);
    case 863:
        return ScrambledRadicalInverseSpecialized<6701>(a, seed);
    case 864:
        return ScrambledRadicalInverseSpecialized<6703>(a, seed);
    case 865:
        return ScrambledRadicalInverseSpecialized<6709>(a, seed);
    case 866:
        return ScrambledRadicalInverseSpecialized<6719>(a, seed);
    case 867:
        return ScrambledRadicalInverseSpecialized<6733>(a, seed);
    case 868:
        return ScrambledRadicalInverseSpecialized<6737>(a, seed);
    case 869:
        return ScrambledRadicalInverseSpecialized<6761>(a, seed);
    case 870:
        return ScrambledRadicalInverseSpecialized<6763>(a, seed);
    case 871:
        return ScrambledRadicalInverseSpecialized<6779>(a, seed);
    case 872:
        return ScrambledRadicalInverseSpecialized<6781>(a, seed);
    case 873:
        return ScrambledRadicalInverseSpecialized<6791>(a, seed);
    case 874:
        return ScrambledRadicalInverseSpecialized<6793>(a, seed);
    case 875:
        return ScrambledRadicalInverseSpecialized<6803>(a, seed);
    case 876:
        return ScrambledRadicalInverseSpecialized<6823>(a, seed);
    case 877:
        return ScrambledRadicalInverseSpecialized<6827>(a, seed);
    case 878:
        return ScrambledRadicalInverseSpecialized<6829>(a, seed);
    case 879:
        return ScrambledRadicalInverseSpecialized<6833>(a, seed);
    case 880:
        return ScrambledRadicalInverseSpecialized<6841>(a, seed);
    case 881:
        return ScrambledRadicalInverseSpecialized<6857>(a, seed);
    case 882:
        return ScrambledRadicalInverseSpecialized<6863>(a, seed);
    case 883:
        return ScrambledRadicalInverseSpecialized<6869>(a, seed);
    case 884:
        return ScrambledRadicalInverseSpecialized<6871>(a, seed);
    case 885:
        return ScrambledRadicalInverseSpecialized<6883>(a, seed);
    case 886:
        return ScrambledRadicalInverseSpecialized<6899>(a, seed);
    case 887:
        return ScrambledRadicalInverseSpecialized<6907>(a, seed);
    case 888:
        return ScrambledRadicalInverseSpecialized<6911>(a, seed);
    case 889:
        return ScrambledRadicalInverseSpecialized<6917>(a, seed);
    case 890:
        return ScrambledRadicalInverseSpecialized<6947>(a, seed);
    case 891:
        return ScrambledRadicalInverseSpecialized<6949>(a, seed);
    case 892:
        return ScrambledRadicalInverseSpecialized<6959>(a, seed);
    case 893:
        return ScrambledRadicalInverseSpecialized<6961>(a, seed);
    case 894:
        return ScrambledRadicalInverseSpecialized<6967>(a, seed);
    case 895:
        return ScrambledRadicalInverseSpecialized<6971>(a, seed);
    case 896:
        return ScrambledRadicalInverseSpecialized<6977>(a, seed);
    case 897:
        return ScrambledRadicalInverseSpecialized<6983>(a, seed);
    case 898:
        return ScrambledRadicalInverseSpecialized<6991>(a, seed);
    case 899:
        return ScrambledRadicalInverseSpecialized<6997>(a, seed);
    case 900:
        return ScrambledRadicalInverseSpecialized<7001>(a, seed);
    case 901:
        return ScrambledRadicalInverseSpecialized<7013>(a, seed);
    case 902:
        return ScrambledRadicalInverseSpecialized<7019>(a, seed);
    case 903:
        return ScrambledRadicalInverseSpecialized<7027>(a, seed);
    case 904:
        return ScrambledRadicalInverseSpecialized<7039>(a, seed);
    case 905:
        return ScrambledRadicalInverseSpecialized<7043>(a, seed);
    case 906:
        return ScrambledRadicalInverseSpecialized<7057>(a, seed);
    case 907:
        return ScrambledRadicalInverseSpecialized<7069>(a, seed);
    case 908:
        return ScrambledRadicalInverseSpecialized<7079>(a, seed);
    case 909:
        return ScrambledRadicalInverseSpecialized<7103>(a, seed);
    case 910:
        return ScrambledRadicalInverseSpecialized<7109>(a, seed);
    case 911:
        return ScrambledRadicalInverseSpecialized<7121>(a, seed);
    case 912:
        return ScrambledRadicalInverseSpecialized<7127>(a, seed);
    case 913:
        return ScrambledRadicalInverseSpecialized<7129>(a, seed);
    case 914:
        return ScrambledRadicalInverseSpecialized<7151>(a, seed);
    case 915:
        return ScrambledRadicalInverseSpecialized<7159>(a, seed);
    case 916:
        return ScrambledRadicalInverseSpecialized<7177>(a, seed);
    case 917:
        return ScrambledRadicalInverseSpecialized<7187>(a, seed);
    case 918:
        return ScrambledRadicalInverseSpecialized<7193>(a, seed);
    case 919:
        return ScrambledRadicalInverseSpecialized<7207>(a, seed);
    case 920:
        return ScrambledRadicalInverseSpecialized<7211>(a, seed);
    case 921:
        return ScrambledRadicalInverseSpecialized<7213>(a, seed);
    case 922:
        return ScrambledRadicalInverseSpecialized<7219>(a, seed);
    case 923:
        return ScrambledRadicalInverseSpecialized<7229>(a, seed);
    case 924:
        return ScrambledRadicalInverseSpecialized<7237>(a, seed);
    case 925:
        return ScrambledRadicalInverseSpecialized<7243>(a, seed);
    case 926:
        return ScrambledRadicalInverseSpecialized<7247>(a, seed);
    case 927:
        return ScrambledRadicalInverseSpecialized<7253>(a, seed);
    case 928:
        return ScrambledRadicalInverseSpecialized<7283>(a, seed);
    case 929:
        return ScrambledRadicalInverseSpecialized<7297>(a, seed);
    case 930:
        return ScrambledRadicalInverseSpecialized<7307>(a, seed);
    case 931:
        return ScrambledRadicalInverseSpecialized<7309>(a, seed);
    case 932:
        return ScrambledRadicalInverseSpecialized<7321>(a, seed);
    case 933:
        return ScrambledRadicalInverseSpecialized<7331>(a, seed);
    case 934:
        return ScrambledRadicalInverseSpecialized<7333>(a, seed);
    case 935:
        return ScrambledRadicalInverseSpecialized<7349>(a, seed);
    case 936:
        return ScrambledRadicalInverseSpecialized<7351>(a, seed);
    case 937:
        return ScrambledRadicalInverseSpecialized<7369>(a, seed);
    case 938:
        return ScrambledRadicalInverseSpecialized<7393>(a, seed);
    case 939:
        return ScrambledRadicalInverseSpecialized<7411>(a, seed);
    case 940:
        return ScrambledRadicalInverseSpecialized<7417>(a, seed);
    case 941:
        return ScrambledRadicalInverseSpecialized<7433>(a, seed);
    case 942:
        return ScrambledRadicalInverseSpecialized<7451>(a, seed);
    case 943:
        return ScrambledRadicalInverseSpecialized<7457>(a, seed);
    case 944:
        return ScrambledRadicalInverseSpecialized<7459>(a, seed);
    case 945:
        return ScrambledRadicalInverseSpecialized<7477>(a, seed);
    case 946:
        return ScrambledRadicalInverseSpecialized<7481>(a, seed);
    case 947:
        return ScrambledRadicalInverseSpecialized<7487>(a, seed);
    case 948:
        return ScrambledRadicalInverseSpecialized<7489>(a, seed);
    case 949:
        return ScrambledRadicalInverseSpecialized<7499>(a, seed);
    case 950:
        return ScrambledRadicalInverseSpecialized<7507>(a, seed);
    case 951:
        return ScrambledRadicalInverseSpecialized<7517>(a, seed);
    case 952:
        return ScrambledRadicalInverseSpecialized<7523>(a, seed);
    case 953:
        return ScrambledRadicalInverseSpecialized<7529>(a, seed);
    case 954:
        return ScrambledRadicalInverseSpecialized<7537>(a, seed);
    case 955:
        return ScrambledRadicalInverseSpecialized<7541>(a, seed);
    case 956:
        return ScrambledRadicalInverseSpecialized<7547>(a, seed);
    case 957:
        return ScrambledRadicalInverseSpecialized<7549>(a, seed);
    case 958:
        return ScrambledRadicalInverseSpecialized<7559>(a, seed);
    case 959:
        return ScrambledRadicalInverseSpecialized<7561>(a, seed);
    case 960:
        return ScrambledRadicalInverseSpecialized<7573>(a, seed);
    case 961:
        return ScrambledRadicalInverseSpecialized<7577>(a, seed);
    case 962:
        return ScrambledRadicalInverseSpecialized<7583>(a, seed);
    case 963:
        return ScrambledRadicalInverseSpecialized<7589>(a, seed);
    case 964:
        return ScrambledRadicalInverseSpecialized<7591>(a, seed);
    case 965:
        return ScrambledRadicalInverseSpecialized<7603>(a, seed);
    case 966:
        return ScrambledRadicalInverseSpecialized<7607>(a, seed);
    case 967:
        return ScrambledRadicalInverseSpecialized<7621>(a, seed);
    case 968:
        return ScrambledRadicalInverseSpecialized<7639>(a, seed);
    case 969:
        return ScrambledRadicalInverseSpecialized<7643>(a, seed);
    case 970:
        return ScrambledRadicalInverseSpecialized<7649>(a, seed);
    case 971:
        return ScrambledRadicalInverseSpecialized<7669>(a, seed);
    case 972:
        return ScrambledRadicalInverseSpecialized<7673>(a, seed);
    case 973:
        return ScrambledRadicalInverseSpecialized<7681>(a, seed);
    case 974:
        return ScrambledRadicalInverseSpecialized<7687>(a, seed);
    case 975:
        return ScrambledRadicalInverseSpecialized<7691>(a, seed);
    case 976:
        return ScrambledRadicalInverseSpecialized<7699>(a, seed);
    case 977:
        return ScrambledRadicalInverseSpecialized<7703>(a, seed);
    case 978:
        return ScrambledRadicalInverseSpecialized<7717>(a, seed);
    case 979:
        return ScrambledRadicalInverseSpecialized<7723>(a, seed);
    case 980:
        return ScrambledRadicalInverseSpecialized<7727>(a, seed);
    case 981:
        return ScrambledRadicalInverseSpecialized<7741>(a, seed);
    case 982:
        return ScrambledRadicalInverseSpecialized<7753>(a, seed);
    case 983:
        return ScrambledRadicalInverseSpecialized<7757>(a, seed);
    case 984:
        return ScrambledRadicalInverseSpecialized<7759>(a, seed);
    case 985:
        return ScrambledRadicalInverseSpecialized<7789>(a, seed);
    case 986:
        return ScrambledRadicalInverseSpecialized<7793>(a, seed);
    case 987:
        return ScrambledRadicalInverseSpecialized<7817>(a, seed);
    case 988:
        return ScrambledRadicalInverseSpecialized<7823>(a, seed);
    case 989:
        return ScrambledRadicalInverseSpecialized<7829>(a, seed);
    case 990:
        return ScrambledRadicalInverseSpecialized<7841>(a, seed);
    case 991:
        return ScrambledRadicalInverseSpecialized<7853>(a, seed);
    case 992:
        return ScrambledRadicalInverseSpecialized<7867>(a, seed);
    case 993:
        return ScrambledRadicalInverseSpecialized<7873>(a, seed);
    case 994:
        return ScrambledRadicalInverseSpecialized<7877>(a, seed);
    case 995:
        return ScrambledRadicalInverseSpecialized<7879>(a, seed);
    case 996:
        return ScrambledRadicalInverseSpecialized<7883>(a, seed);
    case 997:
        return ScrambledRadicalInverseSpecialized<7901>(a, seed);
    case 998:
        return ScrambledRadicalInverseSpecialized<7907>(a, seed);
    case 999:
        return ScrambledRadicalInverseSpecialized<7919>(a, seed);
    case 1000:
        return ScrambledRadicalInverseSpecialized<7927>(a, seed);
    case 1001:
        return ScrambledRadicalInverseSpecialized<7933>(a, seed);
    case 1002:
        return ScrambledRadicalInverseSpecialized<7937>(a, seed);
    case 1003:
        return ScrambledRadicalInverseSpecialized<7949>(a, seed);
    case 1004:
        return ScrambledRadicalInverseSpecialized<7951>(a, seed);
    case 1005:
        return ScrambledRadicalInverseSpecialized<7963>(a, seed);
    case 1006:
        return ScrambledRadicalInverseSpecialized<7993>(a, seed);
    case 1007:
        return ScrambledRadicalInverseSpecialized<8009>(a, seed);
    case 1008:
        return ScrambledRadicalInverseSpecialized<8011>(a, seed);
    case 1009:
        return ScrambledRadicalInverseSpecialized<8017>(a, seed);
    case 1010:
        return ScrambledRadicalInverseSpecialized<8039>(a, seed);
    case 1011:
        return ScrambledRadicalInverseSpecialized<8053>(a, seed);
    case 1012:
        return ScrambledRadicalInverseSpecialized<8059>(a, seed);
    case 1013:
        return ScrambledRadicalInverseSpecialized<8069>(a, seed);
    case 1014:
        return ScrambledRadicalInverseSpecialized<8081>(a, seed);
    case 1015:
        return ScrambledRadicalInverseSpecialized<8087>(a, seed);
    case 1016:
        return ScrambledRadicalInverseSpecialized<8089>(a, seed);
    case 1017:
        return ScrambledRadicalInverseSpecialized<8093>(a, seed);
    case 1018:
        return ScrambledRadicalInverseSpecialized<8101>(a, seed);
    case 1019:
        return ScrambledRadicalInverseSpecialized<8111>(a, seed);
    case 1020:
        return ScrambledRadicalInverseSpecialized<8117>(a, seed);
    case 1021:
        return ScrambledRadicalInverseSpecialized<8123>(a, seed);
    case 1022:
        return ScrambledRadicalInverseSpecialized<8147>(a, seed);
    case 1023:
        return ScrambledRadicalInverseSpecialized<8161>(a, seed);
    default:
        LOG_FATAL("Base %d is >= 1024, the limit of ScrambledRadicalInverse", baseIndex);
        return 0;
    }
}

}  // namespace pbrt
