package llama

import (
	"strings"
	"testing"
)

// A bridge result carries model-produced bytes twice: as a JSON string and as
// b64. The b64 copy is authoritative because a JSON string cannot carry a
// partial UTF-8 sequence losslessly; text alone must still work for bridges
// that predate the b64 field.
func TestTextFieldsBytes(t *testing.T) {
	// 0xE7 0xAB is the first two bytes of a three-byte character: the kind of
	// piece a byte-level tokenizer produces when a character spans tokens.
	partial := "\xe7\xab"
	cases := []struct {
		name    string
		in      textFields
		want    string
		wantErr string
	}{
		{name: "text only", in: textFields{Text: "hello"}, want: "hello"},
		{name: "b64 wins", in: textFields{Text: "\ufffd\ufffd", B64: "56s="}, want: partial},
		{name: "b64 empty text", in: textFields{Text: "", B64: "aGk="}, want: "hi"},
		{name: "bad b64", in: textFields{Text: "x", B64: "!!"}, wantErr: "bad b64 payload"},
	}
	for _, tc := range cases {
		got, err := tc.in.bytes("test")
		if tc.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("%s: err = %v, want %q", tc.name, err, tc.wantErr)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}

// decodeText accepts a result with and without the b64 field.
func TestDecodeTextB64(t *testing.T) {
	got, err := decodeText("t", `{"ok":true,"text":"\ufffd","b64":"5w=="}`)
	if err != nil || got != "\xe7" {
		t.Fatalf("with b64: got %q, %v; want the raw byte 0xE7", got, err)
	}
	got, err = decodeText("t", `{"ok":true,"text":"plain"}`)
	if err != nil || got != "plain" {
		t.Fatalf("without b64: got %q, %v", got, err)
	}
}
