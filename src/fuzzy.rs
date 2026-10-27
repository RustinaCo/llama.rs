//! Fuzzy string matching and text normalization for keyboard proximity and typos.
//!
//! Provides Damerau-Levenshtein edit distance, Persian character normalization,
//! and fuzzy keyword extraction. Zero external dependencies.

/// Compute the Damerau-Levenshtein distance between two string slices.
///
/// Accounts for:
/// - Insertion
/// - Deletion
/// - Substitution
/// - Transposition of two adjacent characters (e.g. typing "اب" instead of "با")
pub fn damerau_levenshtein(s1: &str, s2: &str) -> usize {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let mut d = dp[i - 1][j] + 1; // deletion
            d = d.min(dp[i][j - 1] + 1); // insertion
            d = d.min(dp[i - 1][j - 1] + cost); // substitution

            // Transposition check
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d = d.min(dp[i - 2][j - 2] + 1);
            }

            dp[i][j] = d;
        }
    }

    dp[m][n]
}

/// Normalize common Persian / Arabic Unicode variations and non-spacing marks.
pub fn normalize_persian(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            // Zero-width non-joiner / joiner / directional marks -> space
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => out.push(' '),
            // Arabic Yeh / Farsi Yeh / Alef Maksura
            'ي' | 'ى' | 'ئ' => out.push('ی'),
            // Arabic Kaf
            'ك' => out.push('ک'),
            // Arabic / Persian digits normalization to Persian
            '0' | '٠' => out.push('۰'),
            '1' | '١' => out.push('۱'),
            '2' | '٢' => out.push('۲'),
            '3' | '٣' => out.push('۳'),
            '4' | '٤' => out.push('۴'),
            '5' | '٥' => out.push('۵'),
            '6' | '٦' => out.push('۶'),
            '7' | '٧' => out.push('۷'),
            '8' | '٨' => out.push('۸'),
            '9' | '٩' => out.push('۹'),
            // Punctuations
            '?' | '؟' | '!' | '،' | ',' | '.' | ':' | '؛' => out.push(' '),
            other => out.push(other),
        }
    }
    out
}

/// Tokenize and test if any token in `input` is within `max_distance` of `target`.
pub fn has_fuzzy_match(input: &str, target: &str, max_distance: usize) -> bool {
    let norm = normalize_persian(input);
    for token in norm.split_whitespace() {
        if damerau_levenshtein(token, target) <= max_distance {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(damerau_levenshtein("هست", "هست"), 0);
        // 1 typo: هس -> هش
        assert_eq!(damerau_levenshtein("هس", "هش"), 1);
        // 1 transposition: نیست -> نیتس
        assert_eq!(damerau_levenshtein("نیست", "نیتس"), 1);
        // Insertion
        assert_eq!(damerau_levenshtein("مسکو", "مسکوی"), 1);
    }

    #[test]
    fn test_persian_normalization() {
        assert_eq!(normalize_persian("كتاب‌هاي فارسي"), "کتاب های فارسی");
    }

    #[test]
    fn test_fuzzy_match() {
        // "هش" matches "هس" with max_distance 1
        assert!(has_fuzzy_match("پایتخت مسکو هش", "هس", 1));
        // "نیتس" matches "نیست" with max_distance 1
        assert!(has_fuzzy_match("این درست نیتس", "نیست", 1));
        assert!(!has_fuzzy_match("سلام دنیا", "هست", 1));
    }
}
