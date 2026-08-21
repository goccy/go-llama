package llama

// Test-only windows into the instance internals, for the shared-memory tests.

// EngineImageBacked reports whether the instance's engine memory is a
// copy-on-write map of a shared image rather than a private allocation.
func EngineImageBacked(l *Llama) bool { return l.e().ImageBacked() }
