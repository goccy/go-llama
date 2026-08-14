package llama

// Tokenizers are byte-level: a token is a run of BYTES, not of characters, so
// one multi-byte character can arrive split across two tokens. That is normal
// for CJK and emoji, where a single character is three or four bytes and the
// tokenizer may have learned only part of it as a unit.
//
// Nothing downstream expects that. A Go string holding half a character
// prints as U+FFFD, breaks `for range`, and corrupts anything that
// concatenates it with a length prefix. So a piece is only handed to a caller
// once it is a complete sequence; a truncated tail waits for the token that
// finishes it. llama.cpp's own server does the same thing (validate_utf8 in
// tools/server), for the same reason.

// leadByteLen returns the length of the UTF-8 sequence a lead byte starts, or
// 0 for a continuation byte. This is UTF-8's own encoding rule, read straight
// off the byte:
//
//	0xxxxxxx  1 byte     110xxxxx  2 bytes
//	10xxxxxx  continuation         1110xxxx  3 bytes
//	                               11110xxx  4 bytes
//
// An invalid lead byte (0xF8..0xFF) reports 1, so it passes through as its own
// (broken) byte rather than making the caller wait for bytes that will never
// come.
func leadByteLen(c byte) int {
	switch {
	case c&0x80 == 0x00:
		return 1
	case c&0xC0 == 0x80:
		return 0 // continuation byte, not a start
	case c&0xE0 == 0xC0:
		return 2
	case c&0xF0 == 0xE0:
		return 3
	case c&0xF8 == 0xF0:
		return 4
	}
	return 1
}

// completeUTF8Prefix returns the length of the longest prefix of b that does
// not end part-way through a multi-byte sequence. The returned length is
// len(b) unless the last few bytes are a sequence still waiting for its
// continuation bytes.
//
// It looks back at most four bytes, which is the longest a UTF-8 sequence can
// be: further back is by definition already complete. Bytes that are not
// valid UTF-8 at all are never held back — they cannot be completed, so
// holding them would stall the stream forever.
func completeUTF8Prefix(b []byte) int {
	for i := 1; i <= 4 && i <= len(b); i++ {
		n := leadByteLen(b[len(b)-i])
		if n == 0 {
			continue // continuation byte: keep walking back to the lead
		}
		if n > i {
			return len(b) - i // truncated sequence: hold it back
		}
		return len(b)
	}
	return len(b)
}
