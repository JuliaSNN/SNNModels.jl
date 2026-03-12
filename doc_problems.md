# SNNModels Documentation Review - Problems and Issues

This document tracks documentation issues and functions that need refactoring during the SNNModels library review.

## Overview

Total source files: 73
Review approach: File-by-file systematic documentation and test creation

## Documentation Issues

### Missing Documentation

#### utils/io.jl
- `SNNfolder` - needs documentation
- `SNNfile` - needs documentation  
- `SNNpath` - needs documentation
- `SNNsave` - needs documentation
- Several other save/load helper functions

#### utils/structs.jl
- `isa_model` - needs documentation
- `validate_population_model` - needs documentation
- `validate_synapse_model` - needs documentation
- `validate_stimulus_model` - needs documentation

### Incorrect/Outdated Documentation
None identified yet.

### Unclear Documentation
None identified yet.

## Code Quality Issues

### Functions Too Complex/Convolved

#### utils/util.jl
- `compose()` function is quite complex and handles multiple responsibilities (merging models, extracting items, sorting). Consider splitting into:
  - `extract_model_components()` - for extraction logic
  - `merge_model_components()` - for merging logic
  - `compose()` - as the main coordinator

- `print_model()` function does both graph analysis and printing. Could be split into:
  - `analyze_model_structure()` - returns structured data
  - `format_model_info()` - formats for display
  - `print_model()` - coordinates both

- `extract_items()` is recursive and handles many cases. The nesting logic could be simplified or documented better.

### Naming Issues
None identified yet.

### Other Issues
None identified yet.

## Review Progress

### Phase 1: Documentation Review

#### utils/ (10 files)
- [x] utils/macros.jl - SKIPPED (manual review requested by user)
- [x] utils/unit.jl - Has documentation, looks good ✓
- [x] utils/structs.jl - Added documentation to Time(), isa_model(), validate_*() functions ✓
- [x] utils/util.jl - Added documentation to: rand_value, exp32, exp64, exp256, name, str_name, f2l, remove_element ✓
- [x] utils/io.jl - Added comprehensive documentation to all major functions ✓
- [ ] utils/graph.jl - Needs review
- [ ] utils/record.jl - Needs review
- [ ] utils/main.jl - Needs review
- [ ] utils/sparse_matrix.jl - Needs review
- [ ] utils/spatial.jl - Needs review

### Phase 2: Test Creation

#### utils/ tests created
- [x] test/utils/util_test.jl - Tests for rand_value, exp functions, name generation, f2l, compose, remove_element ✓
- [x] test/utils/structs_test.jl - Tests for Time, EmptyParam, model validation functions ✓
- [x] test/utils/io_test.jl - Tests for save/load functions, file path generation ✓
- [ ] test/utils/graph_test.jl - Needs creation
- [ ] test/utils/record_test.jl - Needs creation
- [ ] test/utils/main_test.jl - Needs creation
- [ ] test/utils/sparse_matrix_test.jl - Needs creation
- [ ] test/utils/spatial_test.jl - Needs creation

### Remaining Work

#### High Priority (Core Functionality)
1. **utils/**: 5 files remaining (graph, record, main, sparse_matrix, spatial)
2. **populations/**: ~15-20 files (neuron models like IF, IZ, HH, AdEx, multicompartment)
3. **connections/**: ~12 files (synapses, plasticity rules)
4. **stimuli/**: ~9 files (input types)
5. **analysis/**: ~3 files (spike analysis, population analysis)

#### Estimated Remaining Files
- **Total source files**: 73
- **Completed**: ~5-6 files
- **Remaining**: ~67-68 files

#### Recommended Next Steps
1. Complete remaining utils files (5 files + 5 test files)
2. Document populations/ - Start with most used models:
   - IF models (if.jl, if_extended.jl)
   - IZ model (iz.jl)
   - Rate model (rate.jl)
   - Poisson (poisson.jl)
3. Document connections/ - Focus on:
   - spiking_synapse.jl
   - sparse_plasticity/ directory
4. Document stimuli/ - All files are relatively small
5. Document analysis/ - 3 files only

---

## Summary of Documentation Added

### utils/util.jl
- `rand_value()` - Generate random values in range
- `exp32()`, `exp64()`, `exp256()` - Fast exponential approximations
- `name()`, `str_name()` - Connection name generation
- `f2l()` - Format string to fixed length  
- `remove_element()` - Remove element from model

### utils/structs.jl
- `Time(Number)` - Constructor from numeric value
- `isa_model()` - Validate model structure
- `validate_population_model()` - Validate population
- `validate_synapse_model()` - Validate synapse
- `validate_stimulus_model()` - Validate stimulus

### utils/io.jl
- `SNNfolder()`, `SNNfile()`, `SNNpath()` - Path generation
- `SNNload()`, `SNNsave()` - Core save/load
- `load_model()`, `load_data()`, `save_model()` - Convenience functions
- `load_or_run()` - Load or compute
- `data2model()` - Convert data to model file
- `save_config()`, `write_config()` - Configuration management
- `get_timestamp()`, `get_git_commit_hash()` - Metadata helpers
- `write_value()` - Recursive value writing
- `read_folder()`, `read_folder!()` - Batch loading

---

*Last Updated: 2026-02-10*
*Progress: ~7% complete (5/73 files documented, 3 test files created)*
