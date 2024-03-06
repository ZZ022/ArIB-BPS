#include<vector>
#include<cstring>
#include<iostream>
#include<fstream>
#include<cmath>
using namespace std;

typedef u_int8_t     BYTE;
typedef u_int16_t    U16;
typedef u_int32_t    U32;
typedef u_int64_t    U64;
typedef __uint128_t  U128;

constexpr BYTE  SIZELOG   = 8;
constexpr U16   TABLESIZE = 1 << SIZELOG;
constexpr BYTE  LPSIZE    = 1 << (SIZELOG - 1);
constexpr U128  ONE = 1;
constexpr U128  RANS_L = ONE << 63;
constexpr static U32 BIT_mask[] = {
    0,          1,         3,         7,         0xF,       0x1F,
    0x3F,       0x7F,      0xFF,      0x1FF,     0x3FF,     0x7FF,
    0xFFF,      0x1FFF,    0x3FFF,    0x7FFF,    0xFFFF,    0x1FFFF,
    0x3FFFF,    0x7FFFF,   0xFFFFF,   0x1FFFFF,  0x3FFFFF,  0x7FFFFF,
    0xFFFFFF,   0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF,
    0x3FFFFFFF, 0x7FFFFFFF };

typedef struct {
    short int deltaFindState;
    U16       deltaNbBits;
}symbolTransformElement;

typedef struct {
    symbolTransformElement transformTable[2];
    BYTE                   stateTable[TABLESIZE];
}encodeElement;

typedef struct {
    BYTE symbol;
    BYTE newState;
    BYTE nbBits;
}decodeElement;

struct BCTable {
    decodeElement m_decodeTable[LPSIZE * TABLESIZE];
    encodeElement m_encodeTable[LPSIZE];

    BCTable() {
        for (size_t i = 1; i <= LPSIZE; i++) {
            setTable(i);
        }
    }
    BYTE getStateId(BYTE state, BYTE symbol, U16 lp, U16& nbBits);
    U16  getDeltaNbBits(BYTE symbol, U16 lp);
    void setTable(U16 lp);
};

static const BCTable BCTABLE;

/*
 tail of the bitstream:
 (encoded k), m_tip, m_bitcontainer of BFSE, m_state
*/
class Bitstream
{
private:
    U128  m_bitContainer;
    U16   m_bitNum;
    BYTE  m_range;
    U32   m_mask;
    U64*  m_startPtr;
    U64*  m_ptr;

public:
    Bitstream(size_t size, BYTE range) :m_bitContainer(0), m_bitNum(0), m_range(range), m_mask(BIT_mask[range]){
        m_startPtr = new U64[size];
        m_ptr = m_startPtr - 1;
    }

    Bitstream(vector<U64> bitstream, size_t size, BYTE range) :m_bitContainer(0), m_bitNum(0), m_range(range), m_mask(BIT_mask[range]){
        m_startPtr = new U64[size];
        m_ptr = m_startPtr + bitstream.size() - 1;
        memcpy(m_startPtr, bitstream.data(), bitstream.size()*sizeof(U64));
    }

    Bitstream(vector<U64> bitstream, BYTE range) :m_bitContainer(0), m_range(range), m_mask(BIT_mask[range]){
        size_t const size = bitstream.size();
        m_startPtr = new U64[size];
        memcpy(m_startPtr, bitstream.data(), size * sizeof(U64));
        m_ptr = m_startPtr + size - 1;
    }
    ~Bitstream(){
        m_ptr = nullptr;
        if(m_startPtr){
            delete[] m_startPtr;
        }
    }
    void        flush();
    void        push(U64 value);
    U64         pop();
    void        addBits(BYTE value, U16 nbBits);
    BYTE        readBits(U16 nbBits);
    void        encodeUniform(U16 value);
    U16         decodeUniform();
    void        setPrecision(U16 precision);
    size_t      getBitstreamSize();
    vector<U64> getBitstream();
    size_t      getBitsLength();
    bool        flushState(BYTE state);
    BYTE        getState(bool isNotAligned);
    U128        switchToRans();
};

class MixEncoder {
private:
    encodeElement const* m_encodeTable;
    BYTE                 m_state;
    U128                 m_tip;
    int                  m_mass_bits;
    U128                 m_mask;
    Bitstream            m_bitstream;

    size_t stateLength() const;
    size_t tipLength() const;
    U64     peek() const;
    void    updateTip(uint64_t peeked, uint64_t pmf, uint64_t cdf);

public:
    MixEncoder() = delete;
    MixEncoder(int mass_bits, BYTE range, size_t size) :m_encodeTable(BCTABLE.m_encodeTable), m_state(0),
    m_tip(0), m_mass_bits(mass_bits), m_mask((ONE << mass_bits) - 1), m_bitstream(size, range){
        if (mass_bits < 1 || mass_bits > 63) {
            throw invalid_argument("mass_bits must be in [1, 63]");
        }
    }
    MixEncoder(int mass_bits, BYTE range, vector<U64> bitstream, size_t size) :m_encodeTable(BCTABLE.m_encodeTable), m_state(0),
    m_tip(0), m_mass_bits(mass_bits), m_mask((ONE << mass_bits) - 1), m_bitstream(bitstream, size, range){
        if (mass_bits < 1 || mass_bits > 63) {
            throw invalid_argument("mass_bits must be in [1, 63]");
        }
    }
    ~MixEncoder() {
    }
    void        encodeBin(BYTE isLps, U16 lpID);  
    U16         decodeSymbol(vector<U64> cdfs);
    void        encodeUniform(U16 symbol);
    bool        switchToRans();
    size_t      getInitSize();
    size_t      getStreamLength();
    vector<U64> getStream();
    bool        flushTip();
    bool        flushState();
    void        setPrecision(U16 precision);
};

class MixDecoder {
private:
    decodeElement const* m_decodeTable;
    BYTE                 m_state;
    U128                 m_tip;
    int                  m_mass_bits;
    U128                 m_mask;
    Bitstream            m_bitstream;

    void   popState(bool isNotAligned);
    void   popTip(bool isLargerThan64);
    
public:
    MixDecoder() = delete;
    MixDecoder(int mass_bits, BYTE range, vector<U64> bitstream) :m_decodeTable(BCTABLE.m_decodeTable), m_state(0),
    m_mass_bits(mass_bits), m_mask((ONE << mass_bits) - 1), m_bitstream(bitstream, range){
    }
    ~MixDecoder() {
    }
    BYTE  decodeBin(U16 lpID);
    void  encodeSymbol(U64 pmf, U64 cdf);
    U16   decodeUniform();
    void  switchFromRans(bool isPop);
    void  prepareForDecode(bool isNotAligned, bool isLargerThan64);
    void  setPrecision(U16 precision);
};

BYTE BCTable::getStateId(BYTE const state, BYTE const symbol, U16 const lp, U16& nbBits) {
    BYTE const mp = TABLESIZE - lp;
    BYTE const freq = symbol ? lp : mp;
    BYTE const offset = symbol ? mp : 0;
    double const logfreq = log2((double)freq);
    int    const logfreq_f = floor(logfreq);
    if (freq > 1 << logfreq_f) {
        BYTE newstate = state >> (SIZELOG - logfreq_f - 1);
        BYTE thre = 2 * (freq - (1 << (logfreq_f)));
        if (newstate >= thre) {
            nbBits = SIZELOG - logfreq_f;
            return  (newstate >> 1) + (1 << logfreq_f) + offset - freq;
        }
        else {
            nbBits = SIZELOG - logfreq_f - 1;
            return  newstate + (1 << (1 + logfreq_f)) + offset - freq;
        }
    }
    else {
        nbBits = SIZELOG - logfreq_f;
        return (state >> nbBits) + offset;
    }
}

U16 BCTable::getDeltaNbBits(BYTE symbol, U16 lp) {
    BYTE const mp = TABLESIZE - lp;
    BYTE const freq = symbol ? lp : mp;
    double const logfreq = log2((double)freq);
    BYTE logfreq_f = floor(logfreq);
    if (freq > 1 << logfreq_f) {
        BYTE thre = (freq - (1 << (logfreq_f))) << (SIZELOG - logfreq_f);
        unsigned maxBits = (SIZELOG - logfreq_f) << 8;
        return maxBits - thre;
    }
    else {
        BYTE nbBits = SIZELOG - logfreq_f;
        return (U16)(nbBits << 8);
    }
}

void BCTable::setTable(U16 lp) {
    BYTE const mp = TABLESIZE - lp;
    BYTE const lpId = lp - 1;
    // spread symbols
    {
        float statelsp = 1.0f / lp, statemsp = 1.0f / mp;
        float const steplsp = (float)1.0f / lp;
        float const stepmsp = (float)1.0f / mp;
        for (size_t i = 0; i < TABLESIZE; i++) {
            if (statemsp <= statelsp) {
                m_decodeTable[(lpId << SIZELOG) + i].symbol = 0;
                statemsp += stepmsp;
            }
            else {
                m_decodeTable[(lpId << SIZELOG) + i].symbol = 1;
                statelsp += steplsp;
            }
        }
    }

    BYTE cumul[2] = { 0,mp };
    for (size_t u = 0; u < TABLESIZE; u++) {
        BYTE const s = m_decodeTable[(lpId << SIZELOG) + u].symbol;
        m_encodeTable[lpId].stateTable[cumul[s]++] = u;
    }

    m_encodeTable[lpId].transformTable[0].deltaFindState = -mp;
    m_encodeTable[lpId].transformTable[0].deltaNbBits = getDeltaNbBits(0, lp);
    m_encodeTable[lpId].transformTable[1].deltaFindState = mp - lp;
    m_encodeTable[lpId].transformTable[1].deltaNbBits = getDeltaNbBits(1, lp);

    for (size_t s = 0; s < TABLESIZE; s++) {
        U16 nbBits;
        BYTE id = getStateId(s, 0, lp, nbBits);
        m_decodeTable[(lpId << SIZELOG) + m_encodeTable[lpId].stateTable[id]].nbBits = nbBits;
        m_decodeTable[(lpId << SIZELOG) + m_encodeTable[lpId].stateTable[id]].newState = (s >> nbBits) << nbBits;

        id = getStateId(s, 1, lp, nbBits);
        m_decodeTable[(lpId << SIZELOG) + m_encodeTable[lpId].stateTable[id]].nbBits = nbBits;
        m_decodeTable[(lpId << SIZELOG) + m_encodeTable[lpId].stateTable[id]].newState = (s >> nbBits) << nbBits;
    }
}

void Bitstream::flush() {
    m_ptr++;
    U16 const bitsLeft = m_bitNum - 64;
    U64 const bitsFlush = m_bitContainer >> bitsLeft;
    m_bitNum -= 64;
    m_bitContainer &= BIT_mask[bitsLeft];
    *m_ptr = bitsFlush;
}

void Bitstream::addBits(BYTE value, U16 nbBits) {
    m_bitContainer = (m_bitContainer << nbBits) | (value & BIT_mask[nbBits]);
    m_bitNum += nbBits;
    if (m_bitNum >= 64) {
        flush();
    }
}

BYTE Bitstream::readBits(U16 nbBits) {
    if (m_bitNum < nbBits) {
        m_bitContainer = m_bitContainer | ((U128)*m_ptr << m_bitNum);
        m_ptr--;
        m_bitNum += 64;
    }
    BYTE const bitsRead = m_bitContainer & BIT_mask[nbBits];
    m_bitNum -= nbBits;
    m_bitContainer >>= nbBits;
    return bitsRead;
}

void Bitstream::encodeUniform(U16 value) {
    m_bitContainer = (m_bitContainer << m_range) | (value & m_mask);
    m_bitNum += m_range;
    if (m_bitNum >= 64) {
        flush();
    }
}

U16 Bitstream::decodeUniform() {
    if (m_bitNum < m_range) {
        m_bitContainer = m_bitContainer | ((U128)*m_ptr << m_bitNum);
        m_ptr--;
        m_bitNum += 64;
    }
    U16 const bitsRead = m_bitContainer & m_mask;
    m_bitNum -= m_range;
    m_bitContainer >>= m_range;
    return bitsRead;
}

void Bitstream::setPrecision(U16 precision) {
    m_range = precision;
    m_mask = (1u<<precision) - 1;
}

void Bitstream::push(U64 value) {
    m_ptr++;
    *m_ptr = value;
}

U64 Bitstream::pop() {
    U64 const value = *m_ptr;
    m_ptr--;
    return value;
}

size_t Bitstream::getBitstreamSize()
{
    return m_ptr - m_startPtr + 1;
}

vector<U64> Bitstream::getBitstream() {
    size_t const size = getBitstreamSize();
    vector<U64> bitstream(size);
    memcpy(&bitstream[0], m_startPtr, size * sizeof(U64));
    return bitstream;
}

size_t Bitstream::getBitsLength()
{
    return getBitstreamSize() * 64 + m_bitNum;
}

bool Bitstream::flushState(BYTE state)
{
    if (m_bitNum){
        m_ptr++;
        U64 const out = (U64)(ONE << m_bitNum) | (U64)m_bitContainer;
        *m_ptr = out;
        m_bitContainer = 0;
        m_bitNum = 0;
        m_ptr++;
        *m_ptr = state;
        return true;
    }
    else{
        m_ptr++;
        *m_ptr = state;
        return false;
    }
}

/*
would cause bug if the element is 0, however, the probability is about 1/2^64
*/
U128  Bitstream::switchToRans(){
    U64  const tail = *m_ptr;
    m_ptr --;
    U128 const value = ((U128)*m_ptr << 64) + tail;
    m_ptr --;
    return value;
}

BYTE Bitstream::getState(bool isNotAligned){
    const BYTE state = *m_ptr;
    m_ptr--;
    if (isNotAligned) {
        U64 const value = *m_ptr;
        m_ptr--;
        U16 size = 0;
        U64 value_copy = value;
        while (value_copy != 0) {
            ++size;
            value_copy >>= 1;
        }
        m_bitNum = size - 1;
        m_bitContainer = value % (1ull << (U64)m_bitNum);
    }
    else{
        m_bitNum = 0;
        m_bitContainer = 0;
    }
    return state;
}

size_t MixEncoder::stateLength() const {
    size_t size = 0;
    U128 state_copy = m_state;
    while (state_copy != 0) {
        ++size;
        state_copy >>= 1;
    }
    return size;
}

size_t MixEncoder::tipLength() const {
    size_t size = 0;
        U128 tip_copy = m_tip;
        while (tip_copy != 0) {
            ++size;
            tip_copy >>= 1;
        }
        return size;
}


U64 MixEncoder::peek() const{
    return m_tip & m_mask;
}

void MixEncoder::updateTip(uint64_t peeked, uint64_t pmf, uint64_t cdf) {  
    m_tip = pmf * (m_tip >> m_mass_bits) + peeked - cdf;
    if (m_tip < RANS_L) {
        m_tip = (m_tip << 64) | m_bitstream.pop();
    }
}

void MixEncoder::encodeBin(BYTE isLps, U16 lpID){
    encodeElement const currentElement = m_encodeTable[lpID];
    U16 const nbBits = (m_state + currentElement.transformTable[isLps].deltaNbBits) >> 8;
    m_bitstream.addBits(m_state, nbBits);
    m_state = currentElement.stateTable[((m_state | TABLESIZE) >> nbBits) + currentElement.transformTable[isLps].deltaFindState];
}

U16 MixEncoder::decodeSymbol(vector<U64> cdfs){
    U64 const peeked = peek();
    size_t size = cdfs.size();
    U16 lo = 0, hi = size - 1;
    while (lo < hi) {
        U16 const mid = lo + (hi - lo) / 2;
        if (cdfs[mid] > peeked) {
            hi = mid;
        }
        else {
            lo = mid + 1;
        }
    }
    U16 const s = lo - 1;
    U64 const pmf = cdfs[s+1]-cdfs[s];
    updateTip(peeked, pmf, cdfs[s]);  
    return s;
}

void MixEncoder::encodeUniform(U16 symbol){
    m_bitstream.encodeUniform(symbol);
}

bool MixEncoder::switchToRans(){
    bool isPop = false;
    m_tip = m_bitstream.switchToRans();
    U128 const pmf = (1<<m_mass_bits);
    if (m_tip >= ((RANS_L >> m_mass_bits) << 64) * pmf) {
        m_bitstream.push((U64)m_tip);
        m_tip >>= 64;
        isPop = true;
    }
    m_tip = ((m_tip / pmf) << m_mass_bits) + (m_tip % pmf);
    return isPop;
}

size_t MixEncoder::getInitSize(){
    return m_bitstream.getBitstreamSize()*64;
}

size_t MixEncoder::getStreamLength(){
    return m_bitstream.getBitsLength() + tipLength() + stateLength();
}

vector<U64> MixEncoder::getStream(){
    return m_bitstream.getBitstream();
}

bool MixEncoder::flushTip(){
    size_t const size = tipLength();
    if(size > 64){
        m_bitstream.push((U64)(m_tip >> 64));
        m_bitstream.push((U64)m_tip);
        return true;
    }
    else {
        m_bitstream.push((U64)m_tip);
        return false;
    }
}

bool MixEncoder::flushState(){
    return m_bitstream.flushState(m_state);
}

void MixEncoder::setPrecision(U16 precision){
    m_bitstream.setPrecision(precision);
}

void MixDecoder::popState(bool isNotAligned){
    m_state = m_bitstream.getState(isNotAligned);
}

void MixDecoder::popTip(bool isLargerThan64){
    if(isLargerThan64){
            U64 const tail = m_bitstream.pop();
            m_tip = ((U128)m_bitstream.pop() << 64) | tail;
        }
    else{
        m_tip = m_bitstream.pop();
    }
}

BYTE  MixDecoder::decodeBin(U16 lpID){
    decodeElement const currentElement = m_decodeTable[(lpID << SIZELOG) + m_state];
    m_state = currentElement.newState + m_bitstream.readBits(currentElement.nbBits);
    return currentElement.symbol;
}

void MixDecoder::encodeSymbol(U64 pmf, U64 cdf){
    if (m_tip >= ((RANS_L >> m_mass_bits) << 64) * pmf) {
        m_bitstream.push((U64)m_tip);
        m_tip >>= 64;
    }
    m_tip = ((m_tip / pmf) << m_mass_bits) + (m_tip % pmf) + cdf;
}

U16 MixDecoder::decodeUniform(){
    return m_bitstream.decodeUniform();
}

void MixDecoder::switchFromRans(bool isPop){
    U128 const pmf = 1<<m_mass_bits;
    U64  peeked = m_tip & m_mask;
    m_tip = pmf * (m_tip >> m_mass_bits) + peeked;
    if(isPop){
        m_tip = (m_tip << 64) | m_bitstream.pop();
    }
    m_bitstream.push(m_tip>>64);
    m_bitstream.push((U64)m_tip);
}

void MixDecoder::prepareForDecode(bool isNotAligned, bool isLargerThan64){
    popTip(isLargerThan64);
    popState(isNotAligned);
}

void MixDecoder::setPrecision(U16 precision){
    m_bitstream.setPrecision(precision);
}