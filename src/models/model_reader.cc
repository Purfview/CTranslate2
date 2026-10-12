#include "ctranslate2/models/model_reader.h"

#include "ctranslate2/filesystem.h"

#include <algorithm>
#include <limits>

namespace ctranslate2 {
  namespace models {

    std::unique_ptr<std::istream> ModelReader::get_required_file(const std::string& filename,
                                                                 const bool binary) {
      std::unique_ptr<std::istream> file = get_file(filename, binary);
      if (!file)
        throw std::runtime_error("Unable to open file '" + filename
                                 + "' in model '" + get_model_id() + "'");
      return file;
    }


    ModelFileReader::ModelFileReader(std::string model_dir)
      : _model_dir(std::move(model_dir))
    {
    }

    std::string ModelFileReader::get_model_id() const {
      return _model_dir;
    }

    std::unique_ptr<std::istream> ModelFileReader::get_file(const std::string& filename,
                                                            const bool binary) {
      const std::string path = _model_dir + "/" + filename;
      const std::ios_base::openmode mode = binary ? std::ios_base::binary : std::ios_base::in;
      auto stream = std::make_unique<std::ifstream>(open_file_read(path, mode, /*check=*/false));
      if (!stream || !(*stream))
        return nullptr;
      return stream;
    }


    struct membuf : std::streambuf {
      char* _base;
      size_t _size;
      size_t _window_start = 0;

      size_t position() const {
        return _window_start + static_cast<size_t>(gptr() - eback());
      }

      // Expose at most INT_MAX characters through setg().
      void set_window(size_t pos) {
        const size_t max_get_area =
          static_cast<size_t>(std::numeric_limits<int>::max());

        _window_start = pos;

        const size_t count = std::min(_size - pos, max_get_area);
        char* first = _base + pos;
        setg(first, first, first + count);
      }

      membuf(const char* base, size_t size)
        : _base(const_cast<char*>(base))
        , _size(size)
      {
        set_window(0);
      }

    protected:
      int_type underflow() override {
        if (gptr() < egptr())
          return traits_type::to_int_type(*gptr());

        const size_t pos = position();
        if (pos >= _size)
          return traits_type::eof();

        // The current window is exhausted; expose the next window.
        set_window(pos);
        return traits_type::to_int_type(*gptr());
      }

      int_type pbackfail(int_type ch = traits_type::eof()) override {
        const size_t pos = position();
        if (pos == 0)
          return traits_type::eof();

        const char c = _base[pos - 1];

        if (!traits_type::eq_int_type(ch, traits_type::eof()) &&
            !traits_type::eq(c, traits_type::to_char_type(ch)))
          return traits_type::eof();

        // Allow unget/putback across a window boundary.
        set_window(pos - 1);
        return traits_type::to_int_type(c);
      }

      std::streampos seekoff(std::streamoff off,
                             std::ios_base::seekdir dir,
                             std::ios_base::openmode which) override {
        if (!(which & std::ios_base::in))
          return std::streampos(-1);

        if (_size > static_cast<size_t>(
                      std::numeric_limits<std::streamoff>::max()))
          return std::streampos(-1);

        const std::streamoff end = static_cast<std::streamoff>(_size);
        std::streamoff base;

        if (dir == std::ios_base::beg)
          base = 0;
        else if (dir == std::ios_base::cur)
          base = static_cast<std::streamoff>(position());
        else if (dir == std::ios_base::end)
          base = end;
        else
          return std::streampos(-1);

        // Validate the offset before adding it, avoiding signed overflow.
        if (off < -base || off > end - base)
          return std::streampos(-1);

        const std::streamoff next = base + off;
        set_window(static_cast<size_t>(next));
        return std::streampos(next);
      }

      std::streampos seekpos(std::streampos pos,
                             std::ios_base::openmode which) override {
        return seekoff(static_cast<std::streamoff>(pos),
                       std::ios_base::beg, which);
      }
    };


    struct imemstream : virtual membuf, std::istream {
      imemstream(const char* base, size_t size)
        : membuf(base, size)
        , std::istream(static_cast<std::streambuf*>(this))
      {
      }
    };


    ModelMemoryReader::ModelMemoryReader(std::string model_name)
      : _model_name(std::move(model_name))
    {
    }

    void ModelMemoryReader::register_file(std::string filename, std::string content) {
      _files.emplace(std::move(filename), std::move(content));
    }

    std::string ModelMemoryReader::get_model_id() const {
      return _model_name;
    }

    std::unique_ptr<std::istream> ModelMemoryReader::get_file(const std::string& filename,
                                                              const bool) {
      const auto it = _files.find(filename);

      if (it == _files.end())
        return nullptr;

      const auto& content = it->second;
      return std::make_unique<imemstream>(content.data(), content.size());
    }


    std::shared_ptr<Vocabulary>
    load_vocabulary(ModelReader& model_reader,
                    const std::string& filename,
                    VocabularyInfo vocab_info) {
      std::unique_ptr<std::istream> file;

      file = model_reader.get_file(filename + ".json");
      if (file)
        return std::make_shared<Vocabulary>(Vocabulary::from_json_file(*file, std::move(vocab_info)));

      file = model_reader.get_file(filename + ".txt");
      if (file)
        return std::make_shared<Vocabulary>(Vocabulary::from_text_file(*file, std::move(vocab_info)));

      return nullptr;
    }

  }
}
