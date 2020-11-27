// matrix/kaldi-matrix.cc

// Copyright 2009-2011   Lukas Burget;  Ondrej Glembek;  Go Vivace Inc.;
//                       Microsoft Corporation;  Saarland University;
//                       Yanmin Qian;  Petr Schwarz;  Jan Silovsky;
//                       Haihua Xu
//           2017        Shiyin Kang

// See https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/COPYING
// for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

// Based on https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.cc

#include "matrix/kaldi-matrix.h"
#include "matrix/compressed-matrix.h"

namespace {

template<typename Real>
void assert_matrix_shape(const torch::Tensor &tensor_);

template<>
void assert_matrix_shape<float>(const torch::Tensor &tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 2);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat32);
}

template<>
void assert_matrix_shape<double>(const torch::Tensor &tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 2);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat64);
}

} // namespace

namespace kaldi {

template<typename Real>
MatrixBase<Real>::MatrixBase(torch::Tensor tensor) : tensor_(tensor) {
  assert_matrix_shape<Real>(tensor_);
};

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.cc#L1377-L1418
template<typename Real>
void MatrixBase<Real>::Write(std::ostream &os, bool binary) const {
  if (!os.good()) {
    KALDI_ERR << "Failed to write matrix to stream: stream not good";
  }
  if (binary) {  // Use separate binary and text formats,
    // since in binary mode we need to know if it's float or double.
    std::string my_token = (sizeof(Real) == 4 ? "FM" : "DM");

    WriteToken(os, binary, my_token);
    {
      int32 rows = this->NumRows();  // make the size 32-bit on disk.
      int32 cols = this->NumCols();
      KALDI_ASSERT(this->NumRows() == (MatrixIndexT) rows);
      KALDI_ASSERT(this->NumCols() == (MatrixIndexT) cols);
      WriteBasicType(os, binary, rows);
      WriteBasicType(os, binary, cols);
    }
    if (Stride() == NumCols())
      os.write(reinterpret_cast<const char*> (Data()), sizeof(Real)
               * static_cast<size_t>(NumRows()) * static_cast<size_t>(NumCols()));
    else
      for (MatrixIndexT i = 0; i < NumRows(); i++)
        os.write(reinterpret_cast<const char*> (RowData(i)), sizeof(Real)
                 * NumCols());
    if (!os.good()) {
      KALDI_ERR << "Failed to write matrix to stream";
    }
  } else {  // text mode.
    if (NumCols() == 0) {
      os << " [ ]\n";
    } else {
      os << " [";
      for (MatrixIndexT i = 0; i < NumRows(); i++) {
        os << "\n  ";
        for (MatrixIndexT j = 0; j < NumCols(); j++)
          os << (*this)(i, j) << " ";
      }
      os << "]\n";
    }
  }
}

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.cc#L1421-L1445
template<typename Real>
void MatrixBase<Real>::Read(std::istream & is, bool binary, bool add) {
  if (add) {
    Matrix<Real> tmp(NumRows(), NumCols());
    tmp.Read(is, binary, false);  // read without adding.
    if (tmp.NumRows() != this->NumRows() || tmp.NumCols() != this->NumCols())
      KALDI_ERR << "MatrixBase::Read, size mismatch "
                << this->NumRows() << ", " << this->NumCols()
                << " vs. " << tmp.NumRows() << ", " << tmp.NumCols();
    this->AddMat(1.0, tmp);
    return;
  }
  // now assume add == false.

  //  In order to avoid rewriting this, we just declare a Matrix and
  // use it to read the data, then copy.
  Matrix<Real> tmp;
  tmp.Read(is, binary, false);
  if (tmp.NumRows() != NumRows() || tmp.NumCols() != NumCols()) {
    KALDI_ERR << "MatrixBase<Real>::Read, size mismatch "
              << NumRows() << " x " << NumCols() << " versus "
              << tmp.NumRows() << " x " << tmp.NumCols();
  }
  CopyFromMat(tmp);
}

// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.cc#L1448-L1619
template<typename Real>
void Matrix<Real>::Read(std::istream & is, bool binary, bool add) {
  if (add) {
    Matrix<Real> tmp;
    tmp.Read(is, binary, false);  // read without adding.
    if (this->NumRows() == 0) this->Resize(tmp.NumRows(), tmp.NumCols());
    else {
      if (this->NumRows() != tmp.NumRows() || this->NumCols() != tmp.NumCols()) {
        if (tmp.NumRows() == 0) return;  // do nothing in this case.
        else KALDI_ERR << "Matrix::Read, size mismatch "
                       << this->NumRows() <<  ", " << this->NumCols()
                       << " vs. " << tmp.NumRows() << ", " << tmp.NumCols();
      }
    }
    this->AddMat(1.0, tmp);
    return;
  }

  // now assume add == false.
  MatrixIndexT pos_at_start = is.tellg();
  std::ostringstream specific_error;

  if (binary) {  // Read in binary mode.
    int peekval = Peek(is, binary);
    if (peekval == 'C') {
      // This code enable us to read CompressedMatrix as a regular matrix.
      CompressedMatrix compressed_mat;
      compressed_mat.Read(is, binary); // at this point, add == false.
      this->Resize(compressed_mat.NumRows(), compressed_mat.NumCols());
      compressed_mat.CopyToMat(this);
      return;
    }
    const char *my_token =  (sizeof(Real) == 4 ? "FM" : "DM");
    char other_token_start = (sizeof(Real) == 4 ? 'D' : 'F');
    if (peekval == other_token_start) {  // need to instantiate the other type to read it.
      typedef typename OtherReal<Real>::Real OtherType;  // if Real == float, OtherType == double, and vice versa.
      Matrix<OtherType> other(this->NumRows(), this->NumCols());
      other.Read(is, binary, false);  // add is false at this point anyway.
      this->Resize(other.NumRows(), other.NumCols());
      this->CopyFromMat(other);
      return;
    }
    std::string token;
    ReadToken(is, binary, &token);
    if (token != my_token) {
      if (token.length() > 20) token = token.substr(0, 17) + "...";
      specific_error << ": Expected token " << my_token << ", got " << token;
      goto bad;
    }
    int32 rows, cols;
    ReadBasicType(is, binary, &rows);  // throws on error.
    ReadBasicType(is, binary, &cols);  // throws on error.
    if ((MatrixIndexT)rows != this->NumRows() || (MatrixIndexT)cols != this->NumCols()) {
      this->Resize(rows, cols);
    }
    if (this->Stride() == this->NumCols() && rows*cols!=0) {
      is.read(reinterpret_cast<char*>(this->Data()),
              sizeof(Real)*rows*cols);
      if (is.fail()) goto bad;
    } else {
      for (MatrixIndexT i = 0; i < (MatrixIndexT)rows; i++) {
        is.read(reinterpret_cast<char*>(this->RowData(i)), sizeof(Real)*cols);
        if (is.fail()) goto bad;
      }
    }
    if (is.eof()) return;
    if (is.fail()) goto bad;
    return;
  } else {  // Text mode.
    std::string str;
    is >> str; // get a token
    if (is.fail()) { specific_error << ": Expected \"[\", got EOF"; goto bad; }
    // if ((str.compare("DM") == 0) || (str.compare("FM") == 0)) {  // Back compatibility.
    // is >> str;  // get #rows
    //  is >> str;  // get #cols
    //  is >> str;  // get "["
    // }
    if (str == "[]") { Resize(0, 0); return; } // Be tolerant of variants.
    else if (str != "[") {
      if (str.length() > 20) str = str.substr(0, 17) + "...";
      specific_error << ": Expected \"[\", got \"" << str << '"';
      goto bad;
    }
    // At this point, we have read "[".
    std::vector<std::vector<Real>* > data;
    std::vector<Real> *cur_row = new std::vector<Real>;
    while (1) {
      int i = is.peek();
      if (i == -1) { specific_error << "Got EOF while reading matrix data"; goto cleanup; }
      else if (static_cast<char>(i) == ']') {  // Finished reading matrix.
        is.get();  // eat the "]".
        i = is.peek();
        if (static_cast<char>(i) == '\r') {
          is.get();
          is.get();  // get \r\n (must eat what we wrote)
        } else if (static_cast<char>(i) == '\n') { is.get(); } // get \n (must eat what we wrote)
        if (is.fail()) {
          KALDI_WARN << "After end of matrix data, read error.";
          // we got the data we needed, so just warn for this error.
        }
        // Now process the data.
        if (!cur_row->empty()) data.push_back(cur_row);
        else delete(cur_row);
        cur_row = NULL;
        if (data.empty()) { this->Resize(0, 0); return; }
        else {
          int32 num_rows = data.size(), num_cols = data[0]->size();
          this->Resize(num_rows, num_cols);
          for (int32 i = 0; i < num_rows; i++) {
            if (static_cast<int32>(data[i]->size()) != num_cols) {
              specific_error << "Matrix has inconsistent #cols: " << num_cols
                             << " vs." << data[i]->size() << " (processing row"
                             << i << ")";
              goto cleanup;
            }
            for (int32 j = 0; j < num_cols; j++)
              (*this)(i, j) = (*(data[i]))[j];
            delete data[i];
            data[i] = NULL;
          }
        }
        return;
      } else if (static_cast<char>(i) == '\n' || static_cast<char>(i) == ';') {
        // End of matrix row.
        is.get();
        if (cur_row->size() != 0) {
          data.push_back(cur_row);
          cur_row = new std::vector<Real>;
          cur_row->reserve(data.back()->size());
        }
      } else if ( (i >= '0' && i <= '9') || i == '-' ) {  // A number...
        Real r;
        is >> r;
        if (is.fail()) {
          specific_error << "Stream failure/EOF while reading matrix data.";
          goto cleanup;
        }
        cur_row->push_back(r);
      } else if (isspace(i)) {
        is.get();  // eat the space and do nothing.
      } else {  // NaN or inf or error.
        std::string str;
        is >> str;
        if (!KALDI_STRCASECMP(str.c_str(), "inf") ||
            !KALDI_STRCASECMP(str.c_str(), "infinity")) {
          cur_row->push_back(std::numeric_limits<Real>::infinity());
          KALDI_WARN << "Reading infinite value into matrix.";
        } else if (!KALDI_STRCASECMP(str.c_str(), "nan")) {
          cur_row->push_back(std::numeric_limits<Real>::quiet_NaN());
          KALDI_WARN << "Reading NaN value into matrix.";
        } else {
          if (str.length() > 20) str = str.substr(0, 17) + "...";
          specific_error << "Expecting numeric matrix data, got " << str;
          goto cleanup;
        }
      }
    }
    // Note, we never leave the while () loop before this
    // line (we return from it.)
 cleanup: // We only reach here in case of error in the while loop above.
    if(cur_row != NULL)
      delete cur_row;
    for (size_t i = 0; i < data.size(); i++)
      if(data[i] != NULL)
        delete data[i];
    // and then go on to "bad" below, where we print error.
  }
bad:
  KALDI_ERR << "Failed to read matrix from stream.  " << specific_error.str()
            << " File position at start is "
            << pos_at_start << ", currently " << is.tellg();
}
  
// https://github.com/kaldi-asr/kaldi/blob/7fb716aa0f56480af31514c7e362db5c9f787fd4/src/matrix/kaldi-matrix.cc#L2060-L2064
template<class Real>
Matrix<Real>::Matrix(const CompressedMatrix &M): MatrixBase<Real>() {
  Resize(M.NumRows(), M.NumCols(), kUndefined);
  M.CopyToMat(this);
}

template struct Matrix<float>;
template struct Matrix<double>;
template struct MatrixBase<float>;
template struct MatrixBase<double>;
template struct SubMatrix<float>;
template struct SubMatrix<double>;

} // namespace kaldi
