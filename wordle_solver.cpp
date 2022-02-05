#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <ranges>
#include <unordered_map>
#include <stdexcept>
#include <utility>
#include <vector>

using Word = std::array<char, 5>;

std::ostream& operator<<(std::ostream& os, Word const& w) {
    for (char const c : w) {
        os << static_cast<char>(c + 'a');
    }

    return os;
}

std::istream& operator>>(std::istream& is, Word& w) {
    for (char& c : w) {
        is >> c;

        if (c < 'a' || c > 'z') {
            throw std::invalid_argument("Tried to read in word with letter outside of a-z range!");
        }

        c -= 'a';
    }

    return is;
}

// This represents the information that was obtained from "guess". Instead of storing the colored squares, we use a
// representation which allows us to test whether another word matches this information very efficiently.
struct WordInfo {
    Word guess;
    std::array<bool, 5> correct_letters;
    std::array<char, 26> min_counts;
    std::array<char, 26> max_counts;

    WordInfo(Word const guess, Word const truth) : guess{guess} {
        for (std::size_t i = 0; i < 5; ++i) {
            correct_letters[i] = (guess[i] == truth[i]);
        }

        std::array<char, 26> guess_counts{0};
        std::array<char, 26> truth_counts{0};

        for (char const c : guess) {
            ++guess_counts[c];
        }

        for (char const c : truth) {
            ++truth_counts[c];
        }

        for (std::size_t i = 0; i < 26; ++i) {
            if (guess_counts[i] <= truth_counts[i]) {
                min_counts[i] = guess_counts[i];
                max_counts[i] = 5;
            } else {
                min_counts[i] = truth_counts[i];
                max_counts[i] = truth_counts[i];
            }
        }
    }

    WordInfo(Word const guess, std::string const& info) : guess{guess}, min_counts{} {
        std::ranges::fill(max_counts, 5);

        for (std::size_t i = 0; i < 5; ++i) {
            if (info[i] == 'g') {
                correct_letters[i] = true;
                ++min_counts[guess[i]];
                continue;
            }

            correct_letters[i] = false;

            if (info[i] == 'y') {
                ++min_counts[guess[i]];
            }
        }

        for (std::size_t i = 0; i < 5; ++i) {
            if (info[i] == 'b') {
                max_counts[guess[i]] = min_counts[guess[i]];
            }
        }
    }

    bool check_word(Word const word) const {
        std::array<char, 26> counts{0};

        for (std::size_t i = 0; i < 5; ++i) {
            if ((word[i] == guess[i]) != correct_letters[i]) {
                return false;
            }

            if (++counts[word[i]] > max_counts[word[i]]) {
                return false;
            }
        }

        return std::ranges::all_of(guess, [&](char const c) { return counts[c] >= min_counts[c]; });
    }

    bool operator==(WordInfo const& other) const = default;
};

// See boost::hash_combine
template <typename T>
void hash_combine(std::size_t& h, T const& c) {
    h ^= std::hash<T>()(c) + 0x9e3779b9 + (h << 6) + (h >> 2);
}

template <>
struct std::hash<Word> {
    std::size_t operator()(Word const& word) const noexcept {
        std::size_t h1 = 0;

        for (std::size_t i = 0; i < 5; ++i) {
            h1 += static_cast<std::size_t>(word[i]) << (i * 8);
        }

        return std::hash<std::size_t>()(h1);
    }
};

// Note: the hash of WordInfo *ignores* the underlying guess. This is because we will only ever compare WordInfos for
// the same guess.
template <>
struct std::hash<WordInfo> {
    std::size_t operator()(WordInfo const& info) const noexcept {
        std::size_t h1 = 0, h2 = 0, h3 = 0;

        for (std::size_t i = 0; i < 5; ++i) {
            h1 += static_cast<std::size_t>(info.correct_letters[i]) << i;
        }

        for (std::size_t i = 0; i < 19; ++i) {
            h1 += static_cast<std::size_t>(info.min_counts[i]) << (i * 3 + 5);
        }

        for (std::size_t i = 19; i < 26; ++i) {
            h2 += static_cast<std::size_t>(info.min_counts[i]) << ((i - 19) * 3);
        }

        for (std::size_t i = 0; i < 13; ++i) {
            h2 += static_cast<std::size_t>(info.max_counts[i]) << ((i + 7) * 3);
        }

        for (std::size_t i = 13; i < 26; ++i) {
            h3 += static_cast<std::size_t>(info.max_counts[i]) << ((i - 13) * 3);
        }

        hash_combine(h1, h2);
        hash_combine(h1, h3);
        return h1;
    }
};

template <typename Fn, typename T>
concept Reduction = requires(Fn&& f, T x, T y) {
    x = std::invoke(f, x, y);
};

// Main function: determine the best word from "allowed_choices" given that we know that only "remaining_words" are
// possible solutions. Ties are broken based on how common we think certain words are ("word_freqs"). The "best" choice
// is assumed to be the one which maximizes some function of the entropy (i.e. log_2(size)) of the remaining words,
// typically either average (random choice) or maximum (adversarial choice).
template <typename Fn>
requires Reduction<Fn, double> std::pair<Word, double> best_choice(std::vector<Word> const& allowed_choices,
                                                                   std::vector<Word> const& remaining_words,
                                                                   std::unordered_map<Word, double> const& word_freqs,
                                                                   Fn&& fn) {
    Word result = allowed_choices.front();
    std::tuple<double, bool, double> objective{std::numeric_limits<double>::infinity(), true, 0.0};
    std::mutex result_mut;

    // We use std parallelization for free performance! Note that we cannot use std::execution::par_unseq because we use
    // a mutex!
    std::for_each(std::execution::par, allowed_choices.begin(), allowed_choices.end(), [&](Word const& guess) {
        double total_entropy = 0.0;
        std::unordered_map<WordInfo, double> memo_map;  // We can memoize entropy computation if we happen to get the
                                                        // same information for different words.

        for (Word const& truth : remaining_words) {
            WordInfo const guess_info{guess, truth};

            if (auto const it = memo_map.find(guess_info); it != memo_map.end()) {
                total_entropy = std::invoke(fn, total_entropy, it->second);
                continue;
            }

            double const entropy = std::log2(std::ranges::count_if(
                remaining_words, [&guess_info, truth](Word const& w) { return guess_info.check_word(w); }));
            total_entropy = std::invoke(fn, total_entropy, entropy);
            memo_map[guess_info] = entropy;

            if (total_entropy > std::get<0>(objective)) {
                break;
            }
        }

        if (total_entropy > std::get<0>(objective)) {
            return;
        }

        // Tiebreakers
        auto const freq_it = word_freqs.find(guess);
        double const freq = freq_it == word_freqs.end() ? 0.0 : freq_it->second;
        std::tuple<double, bool, double> value{total_entropy, !std::ranges::binary_search(remaining_words, guess),
                                               -freq};

        if (value < objective) {
            std::lock_guard<std::mutex> guard(result_mut);

            if (value < objective) {
                result = guess;
                objective = value;
            }
        }
    });

    return {result, std::get<0>(objective)};
}

// Instantiation of best_choice assuming each word from "remainig_words" is equally likely.
std::pair<Word, double> best_choice_avg(std::vector<Word> const& allowed_choices,
                                        std::vector<Word> const& remaining_words,
                                        std::unordered_map<Word, double> const& word_freqs = {}) {
    auto [word, entropy] = best_choice(allowed_choices, remaining_words, word_freqs, std::plus<double>());
    return {word, entropy / remaining_words.size()};
}

// Instantiation of best_choice assuming the correct word from "remainig_words" is chosen adversarially.
std::pair<Word, double> best_choice_adv(std::vector<Word> const& allowed_choices,
                                        std::vector<Word> const& remaining_words,
                                        std::unordered_map<Word, double> const& word_freqs = {}) {
    return best_choice(allowed_choices, remaining_words, word_freqs,
                       [](double const x, double const y) { return std::max(x, y); });
}

std::vector<Word> load_word_list(std::string const& filename) {
    std::vector<Word> result;
    std::ifstream file{filename};

    while (file.good()) {
        Word w;
        file >> w;
        file.ignore();

        result.push_back(w);
    }

    std::ranges::sort(result);

    return result;
}

std::unordered_map<Word, double> load_freq_data(std::string const& filename) {
    std::unordered_map<Word, double> result;
    std::ifstream freq_data_file{filename};

    while (freq_data_file.good()) {
        Word w;
        double f;
        freq_data_file >> w >> f;

        result[w] = f;
    }

    return result;
}

int main(int const argc, char const* const* const argv) {
    if (argc < 3 || argc > 6) {
        std::cout << "Usage: ./wordle_solver guess_list.txt word_list.txt "
                     "[hard mode = 0/1] [adversarial = 0/1] [freq_data.txt]\n";
        return 0;
    }

    // Load guess list
    std::vector<Word> guess_list = load_word_list(argv[1]);
    std::cout << "Loaded guess list with " << guess_list.size() << " words!\n";

    // Load list of possible correct words
    std::vector<Word> word_list = load_word_list(argv[2]);
    std::cout << "Loaded word list with " << word_list.size() << " words!\n";

    // Hard mode: only allow guesses that conform to previous information
    bool const hard_mode = argc >= 4 && std::atoi(argv[3]) > 0;
    // Adversarial: assume correct word is being changed adversarially
    bool const adverserial = argc >= 5 && std::atoi(argv[4]) > 0;

    // Load list of word frequency information for tie breaker
    std::unordered_map<Word, double> freq_data;
    if (argc == 6) {
        freq_data = load_freq_data(argv[5]);
        std::cout << "Loaded word frequency data for " << freq_data.size() << " words!\n";
    }

    while (true) {
        auto const st = std::chrono::high_resolution_clock::now();
        auto [guess, entropy] =
            adverserial ? best_choice_adv(guess_list, word_list) : best_choice_avg(guess_list, word_list, freq_data);
        auto const ct = std::chrono::high_resolution_clock::now();

        std::cout << "Best guess is \"" << guess << "\" with " << (adverserial ? "maximum" : "average") << " entropy "
                  << entropy << ".\n";
        std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(ct - st).count()
                  << " ms.\n";

        std::cout << "Response (b|y|g) * 5: ";
        std::string info_string;
        std::cin >> info_string;

        if (info_string == "ggggg") {
            break;
        }

        WordInfo const info{guess, info_string};

        std::erase_if(word_list, [&](Word const w) { return !info.check_word(w); });

        if (hard_mode) {
            std::erase_if(guess_list, [&](Word const w) { return !info.check_word(w); });
        }

        if (word_list.size() < 10) {
            std::cout << "Remaining words:";

            for (Word const& w : word_list) {
                std::cout << ' ' << w;
            }

            std::cout << '\n';
        }
    }
}
