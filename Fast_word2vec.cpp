#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>

struct WordVectors {
  std::map<std::string, std::vector<float> > wv;
  int vocab_size;
  int vector_length;
};

std::vector<std::string> split(std::string const &input) {
    std::istringstream buffer(input);
    std::vector<std::string> ret;

    std::copy(std::istream_iterator<std::string>(buffer),
              std::istream_iterator<std::string>(),
              std::back_inserter(ret));
    return ret;
}

WordVectors read_word2vec_bin(std::string filename) {
  std::map<std::string, std::vector<float> > vecs;
  std::ifstream fin;
  std::string headerstring;

  fin.open(filename, std::ios::in | std::ios::binary);

  if (!fin) {
    std::cerr << " Error: Couldn't find the file." << "\n";
  }

  std::getline(fin, headerstring);

  std::vector<std::string> header = split(headerstring);

  int vocab_size = std::stoi(header[0]);
  int vector_length = std::stoi(header[1]);

  for (int i = 0; i < vocab_size; i++) {
    std::string word;
    std::vector<float> vec(vector_length);
    while (true) {
      char c;
      fin.read(&c, sizeof(char));
      if (c == 32) {
        break;
      } else if (c != 10) {
        word += c;
      }
      fin.read(reinterpret_cast<char*>(&vec[0]), vector_length*sizeof(float));
      vecs[word] = vec;
    }

  }

  return WordVectors {vecs, vocab_size, vector_length};
}

int main()
{
    WordVectors readvecs = read_word2vec_bin("model/GoogleNews-vectors-negative300.bin");
    return 0;
}
