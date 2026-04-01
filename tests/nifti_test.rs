/// Tests for NIfTI reader and preprocessing.

#[test]
fn test_preprocessing_normalize() {
    let data = vec![0.0, 10.0, 20.0, 30.0, 0.0, 40.0, 50.0, 0.0];
    let normed = brainiac::preprocessing::preprocess(
        &brainiac::NiftiVolume {
            data,
            dims: [2, 2, 2],
            pixdim: [1.0, 1.0, 1.0],
        },
        2,
    );
    // After resize to 2×2×2 and normalize, should have 8 values
    assert_eq!(normed.len(), 8);
}

#[test]
fn test_preprocessing_resize() {
    // 4×4×4 uniform volume → 2×2×2 should preserve value
    let data = vec![42.0f32; 64];
    let resized = brainiac::preprocessing::trilinear_resize(&data, [4, 4, 4], 2);
    assert_eq!(resized.len(), 8);
    for &v in &resized {
        assert!((v - 42.0).abs() < 1e-4, "expected 42.0, got {}", v);
    }
}

#[test]
fn test_preprocessing_resize_downsample() {
    // 8×8×8 gradient → 4×4×4
    let mut data = vec![0.0f32; 512];
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                data[x + y * 8 + z * 64] = (x + y + z) as f32;
            }
        }
    }
    let resized = brainiac::preprocessing::trilinear_resize(&data, [8, 8, 8], 4);
    assert_eq!(resized.len(), 64);
    // Values should be in reasonable range
    for &v in &resized {
        assert!(v >= 0.0 && v <= 21.0, "value {} out of expected range", v);
    }
}
