# include "mix_coder.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

template<typename T>
vector<T> ndarray_to_vector(py::array_t<T>& arr) {
    vector<T> vec(arr.size());
    memcpy(vec.data(), arr.data(), arr.size() * sizeof(T));
    return vec;
}

template<typename T>
py::array_t<T> vector_to_ndarray(vector<T>& vec) {
    py::array_t<T> arr(vec.size());
    memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(T));
    return arr;
}

class PyMixEncoder{
private:
    MixEncoder m_encoder;
public:
    PyMixEncoder() = delete;
    PyMixEncoder(int mass_bits, BYTE range, size_t size):m_encoder(mass_bits, range, size){}
    PyMixEncoder(int mass_bits, BYTE range, py::array_t<U64> bitstream, size_t size):m_encoder(mass_bits, range, ndarray_to_vector<U64>(bitstream), size){}
    ~PyMixEncoder(){}

    void             encodeBins(py::array_t<BYTE> &isLsps, py::array_t<U16> &lpIDs);
    py::array_t<U16> decodeSymbols(py::array_t<U64, py::array::c_style | py::array::forcecast> &cdfss);
    void             encodeUniforms(py::array_t<U16> symbol);
    bool             switchToRans();
    size_t           getInitSize();
    size_t           getStreamLength();
    py::array_t<U64> getStream();
    bool             flushTip();
    bool             flushState();
    void             setPrecision(U16 precision);
};  

class PyMixDecoder{
private:
    MixDecoder m_decoder;
public:
    PyMixDecoder() = delete;
    PyMixDecoder(int mass_bits, BYTE range, py::array_t<U64> bitstream): m_decoder(mass_bits, range, ndarray_to_vector<U64>(bitstream)){}
    ~PyMixDecoder(){}

    py::array_t<BYTE> decodeBins(py::array_t<U16> lpIDs);
    void              encodeSymbols(py::array_t<U64> pmfs, py::array_t<U64> cdfs);
    py::array_t<U16>  decodeUniforms(size_t size);
    void  switchFromRans(bool isPop);
    py::array_t<U64>  getStream();
    void  prepareForDecode(bool isNotAligned, bool isLargerThan64);
    void              setPrecision(U16 precision);
};

void PyMixEncoder::encodeBins(py::array_t<BYTE> &isLsps, py::array_t<U16> &lpIDs){
    vector<BYTE> isLsps_c = ndarray_to_vector<BYTE>(isLsps);
    vector<U16> lpIDs_c = ndarray_to_vector<U16>(lpIDs);
    for (size_t i = 0; i < isLsps_c.size(); i++){
        m_encoder.encodeBin(isLsps_c[i], lpIDs_c[i]);
    }
}

py::array_t<U16> PyMixEncoder::decodeSymbols(py::array_t<U64, py::array::c_style | py::array::forcecast> &cdfss){
    py::buffer_info buf_info = cdfss.request();
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Input array must have 2 dimensions");
    }
    size_t size = buf_info.shape[0];
    size_t symbolSize = buf_info.shape[1];
    vector<U16> symbols(size);
    U64 *ptr = static_cast<U64 *>(buf_info.ptr);
    for (size_t i = 0; i < size; i++){
        vector<U64> cdfs(ptr, ptr + symbolSize);
        symbols[i] = m_encoder.decodeSymbol(cdfs);
        ptr += symbolSize;
    }
    return vector_to_ndarray<U16>(symbols);
}

void PyMixEncoder::encodeUniforms(py::array_t<U16> symbol){
    vector<U16> symbol_c = ndarray_to_vector<U16>(symbol);
    for (size_t i = 0; i < symbol_c.size(); i++){
        m_encoder.encodeUniform(symbol_c[i]);
    }
}

bool PyMixEncoder::switchToRans(){
    return m_encoder.switchToRans();
}

size_t PyMixEncoder::getInitSize(){
    return m_encoder.getInitSize();
}

size_t PyMixEncoder::getStreamLength(){
    return m_encoder.getStreamLength();
}

py::array_t<U64> PyMixEncoder::getStream(){
    auto stream = m_encoder.getStream();
    return vector_to_ndarray<U64>(stream);
}

bool PyMixEncoder::flushTip(){
    return m_encoder.flushTip();
}

bool PyMixEncoder::flushState(){
    return m_encoder.flushState();
}

void PyMixEncoder::setPrecision(U16 precision){
    m_encoder.setPrecision(precision);
}

py::array_t<BYTE> PyMixDecoder::decodeBins(py::array_t<U16> lpIDs){
    vector<U16> lpIDs_c = ndarray_to_vector<U16>(lpIDs);
    vector<BYTE> isLsps_c(lpIDs_c.size());
    for (size_t i = 0; i < lpIDs_c.size(); i++){
        isLsps_c[i] = m_decoder.decodeBin(lpIDs_c[i]);
    }
    return vector_to_ndarray<BYTE>(isLsps_c);
}

void PyMixDecoder::encodeSymbols(py::array_t<U64> pmfs, py::array_t<U64> cdfs){
    vector<U64> pmfs_c = ndarray_to_vector<U64>(pmfs);
    vector<U64> cdfs_c = ndarray_to_vector<U64>(cdfs);
    for (size_t i = 0; i < pmfs_c.size(); i++){
        m_decoder.encodeSymbol(pmfs_c[i], cdfs_c[i]);
    }
}

py::array_t<U16> PyMixDecoder::decodeUniforms(size_t size){
    vector<U16> symbols(size);
    for (size_t i = 0; i < symbols.size(); i++){
        symbols[i] = m_decoder.decodeUniform();
    }
    return vector_to_ndarray<U16>(symbols);
}

void PyMixDecoder::switchFromRans(bool isPop){
    m_decoder.switchFromRans(isPop);
}


void PyMixDecoder::prepareForDecode(bool isNotAligned, bool isLargerThan64){
    m_decoder.prepareForDecode(isNotAligned, isLargerThan64);
}

void PyMixDecoder::setPrecision(U16 precision){
    m_decoder.setPrecision(precision);
}

PYBIND11_MODULE(mixcoder, m){
    py::class_<PyMixEncoder>(m, "MixEncoder")
        .def(py::init<int, BYTE, size_t>())
        .def(py::init<int, BYTE, py::array_t<U64>, size_t>())
        .def("encodeBins", &PyMixEncoder::encodeBins)
        .def("decodeSymbols", &PyMixEncoder::decodeSymbols)
        .def("encodeUniforms", &PyMixEncoder::encodeUniforms)
        .def("switchToRans", &PyMixEncoder::switchToRans)
        .def("getInitSize", &PyMixEncoder::getInitSize)
        .def("getStreamLength", &PyMixEncoder::getStreamLength)
        .def("getStream", &PyMixEncoder::getStream)
        .def("flushTip", &PyMixEncoder::flushTip)
        .def("flushState", &PyMixEncoder::flushState)
        .def("setPrecision", &PyMixEncoder::setPrecision);


    py::class_<PyMixDecoder>(m, "MixDecoder")
        .def(py::init<int, BYTE, py::array_t<U64>>())
        .def("decodeBins", &PyMixDecoder::decodeBins)
        .def("encodeSymbols", &PyMixDecoder::encodeSymbols)
        .def("decodeUniforms", &PyMixDecoder::decodeUniforms)
        .def("switchFromRans", &PyMixDecoder::switchFromRans)
        .def("prepareForDecode", &PyMixDecoder::prepareForDecode)
        .def("setPrecision", &PyMixDecoder::setPrecision);
}




