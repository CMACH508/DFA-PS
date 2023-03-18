#ifndef BOOST_SERIALIZATION_IO_SAVELOAD_EIGEN_H
#define BOOST_SERIALIZATION_IO_SAVELOAD_EIGEN_H

#include <Eigen/Dense>

#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost {
namespace serialization {

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
inline void save(Archive &ar,
                 const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &M,
                 const unsigned int /* file_version */) {
    typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows =
        M.rows();
    typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index cols =
        M.cols();

    ar << rows;
    ar << cols;

    ar << make_array(M.data(), M.size());
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
inline void load(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &M,
                 const unsigned int /* file_version */) {
    typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows, cols;

    ar >> rows;
    ar >> cols;

    // if (rows=!_Rows) throw std::exception(/*"Unexpected number of rows"*/);
    // if (cols=!_Cols) throw std::exception(/*"Unexpected number of cols"*/);

    ar >> make_array(M.data(), M.size());
}

template <class Archive, typename _Scalar, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void load(Archive &ar,
                 Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols, _Options, _MaxRows, _MaxCols> &M,
                 const unsigned int /* file_version */) {
    typename Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols, _Options, _MaxRows, _MaxCols>::Index
        rows,
        cols;

    ar >> rows;
    ar >> cols;

    // if (cols=!_Cols) throw std::exception(/*"Unexpected number of cols"*/);

    M.resize(rows, Eigen::NoChange);

    ar >> make_array(M.data(), M.size());
}

template <class Archive, typename _Scalar, int _Rows, int _Options, int _MaxRows, int _MaxCols>
inline void load(Archive &ar,
                 Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic, _Options, _MaxRows, _MaxCols> &M,
                 const unsigned int /* file_version */) {
    typename Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>::Index
        rows,
        cols;

    ar >> rows;
    ar >> cols;

    // if (rows=!_Rows) throw std::exception(/*"Unexpected number of rows"*/);

    M.resize(Eigen::NoChange, cols);

    ar >> make_array(M.data(), M.size());
}

template <class Archive, typename _Scalar, int _Options, int _MaxRows, int _MaxCols>
inline void
load(Archive &ar,
     Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, _Options, _MaxRows, _MaxCols> &M,
     const unsigned int /* file_version */) {
    typename Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, _Options, _MaxRows,
                           _MaxCols>::Index rows,
        cols;

    ar >> rows;
    ar >> cols;

    M.resize(rows, cols);

    ar >> make_array(M.data(), M.size());
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows,
          int _MaxCols>
inline void serialize(Archive &ar,
                      Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &M,
                      const unsigned int file_version) {
    split_free(ar, M, file_version);
}

template <class Archive, typename _Scalar, int _Dim, int _Mode, int _Options>
inline void serialize(Archive &ar, Eigen::Transform<_Scalar, _Dim, _Mode, _Options> &t,
                      const unsigned int version) {
    serialize(ar, t.matrix(), version);
}

} // namespace serialization
} // namespace boost

#endif // BOOST_SERIALIZATION_IO_SAVELOAD_EIGEN_H
