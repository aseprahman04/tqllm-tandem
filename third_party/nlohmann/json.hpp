#pragma once
// Minimal JSON reader for checkpoint metadata (safetensors headers, index and config files).
#include <cstdint>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tqllm {

// Just enough JSON for safetensors headers and index files: objects, arrays, strings, integers, literals.
struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    std::string str;
    double num = 0;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json * get(const std::string & k) const {
        for (const auto & [key, v] : obj)
            if (key == k) return &v;
        return nullptr;
    }
};

class Parser {
public:
    explicit Parser(const std::string & s) : s_(s) {}
    // Extract a top-level member's original JSON while validating the envelope. Tool
    // arguments may carry integer identifiers wider than the metadata reader's double.
    static Json parse_with_member(const std::string & source, const std::string & key, std::string & raw) {
        Parser parser(source);
        parser.raw_key_ = &key;
        parser.raw_value_ = &raw;
        raw.clear();
        return parser.parse();
    }
    Json parse() {
        Json v = value();
        ws();
        if (i_ != s_.size()) fail("trailing data");
        return v;
    }

private:
    const std::string & s_;
    size_t i_ = 0;
    const std::string * raw_key_ = nullptr;
    std::string * raw_value_ = nullptr;
    int depth_ = 0;  // arrays / objects open; request bodies are untrusted, so recursion is bounded
    static constexpr int kMaxDepth = 512;
    struct Nest {
        Parser & p;
        explicit Nest(Parser & q) : p(q) { if (++p.depth_ > kMaxDepth) p.fail("nesting deeper than 512 levels"); }
        ~Nest() { --p.depth_; }
    };

    [[noreturn]] void fail(const char * what) { throw std::runtime_error(std::string("safetensors JSON: ") + what); }
    void ws() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    char peek() {
        ws();
        if (i_ >= s_.size()) fail("unexpected end");
        return s_[i_];
    }
    void expect(char c) {
        if (peek() != c) fail("unexpected character");
        ++i_;
    }
    uint16_t hex4() {
        if (i_ + 4 > s_.size()) fail("short unicode escape");
        uint16_t value = 0;
        for (int n = 0; n < 4; ++n) {
            const char c = s_[i_++];
            value = static_cast<uint16_t>(value << 4);
            if (c >= '0' && c <= '9') value = static_cast<uint16_t>(value + c - '0');
            else if (c >= 'a' && c <= 'f') value = static_cast<uint16_t>(value + c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value = static_cast<uint16_t>(value + c - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return value;
    }
    void utf8(std::string & out, uint32_t cp) {
        if (cp <= 0x7F) out.push_back(static_cast<char>(cp));
        else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0x10FFFF) {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else fail("invalid unicode code point");
    }
    std::string string() {
        expect('"');
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\') {
                if (++i_ >= s_.size()) fail("bad escape");
                const char e = s_[i_];
                switch (e) {
                    case '"': out.push_back('"'); ++i_; break;
                    case '\\': out.push_back('\\'); ++i_; break;
                    case '/': out.push_back('/'); ++i_; break;
                    case 'b': out.push_back('\b'); ++i_; break;
                    case 'f': out.push_back('\f'); ++i_; break;
                    case 'n': out.push_back('\n'); ++i_; break;
                    case 'r': out.push_back('\r'); ++i_; break;
                    case 't': out.push_back('\t'); ++i_; break;
                    case 'u': {
                        ++i_;
                        const uint16_t first = hex4();
                        uint32_t cp = first;
                        if (first >= 0xD800 && first <= 0xDBFF) {
                            if (i_ + 2 > s_.size() || s_[i_] != '\\' || s_[i_ + 1] != 'u') fail("missing low surrogate");
                            i_ += 2;
                            const uint16_t second = hex4();
                            if (second < 0xDC00 || second > 0xDFFF) fail("invalid low surrogate");
                            cp = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                        } else if (first >= 0xDC00 && first <= 0xDFFF) fail("unexpected low surrogate");
                        utf8(out, cp);
                        break;
                    }
                    default: fail("invalid escape");
                }
            } else {
                if (static_cast<unsigned char>(s_[i_]) < 0x20) fail("control character in string");
                out += s_[i_];
                ++i_;
            }
        }
        expect('"');
        return out;
    }
    Json value() {
        Json v;
        const char c = peek();
        if (c == '{') {
            const Nest nest(*this);
            v.kind = Json::Object;
            ++i_;
            if (peek() == '}') return ++i_, v;
            for (;;) {
                auto k = string();
                expect(':');
                ws();
                const size_t begin = i_;
                auto member = value();
                if (raw_key_ && depth_ == 1 && k == *raw_key_) {
                    if (!raw_value_->empty()) fail("duplicate captured member");
                    *raw_value_ = s_.substr(begin, i_ - begin);
                }
                v.obj.emplace_back(std::move(k), std::move(member));
                if (peek() == ',') { ++i_; continue; }
                expect('}');
                return v;
            }
        }
        if (c == '[') {
            const Nest nest(*this);
            v.kind = Json::Array;
            ++i_;
            if (peek() == ']') return ++i_, v;
            for (;;) {
                v.arr.push_back(value());
                if (peek() == ',') { ++i_; continue; }
                expect(']');
                return v;
            }
        }
        if (c == '"') {
            v.kind = Json::String;
            v.str = string();
            return v;
        }
        if (s_.compare(i_, 4, "true") == 0) return i_ += 4, v.kind = Json::Bool, v.num = 1, v;
        if (s_.compare(i_, 5, "false") == 0) return i_ += 5, v.kind = Json::Bool, v;
        if (s_.compare(i_, 4, "null") == 0) return i_ += 4, v;
        const size_t start = i_;
        auto take = [&](char ch) { if (i_ < s_.size() && s_[i_] == ch) { ++i_; return true; } return false; };
        auto digits = [&] {
            const size_t begin = i_;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
            if (begin == i_) fail("expected digit");
        };
        take('-');
        if (!take('0')) digits();
        if (take('.')) digits();
        if (take('e') || take('E')) {
            if (!take('+')) take('-');
            digits();
        }
        v.kind = Json::Number;
        if (!raw_key_) v.num = std::stod(s_.substr(start, i_ - start));
        return v;
    }
};

inline Json parse_json(const std::string & text) { return Parser(text).parse(); }

} // namespace tqllm
