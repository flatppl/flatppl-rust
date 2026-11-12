module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<5x3xf32>, tensor<2xui64>) {
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %10 = stablehlo.constant dense<"0x5555D53FABAA2A40ABAA6A405555D53FABAA2A40ABAA6A405555D53FABAA2A40ABAA6A405555D53FABAA2A40ABAA6A405555D53FABAA2A40ABAA6A40"> : tensor<15xf32>
    %15 = stablehlo.constant dense<"0xA532843EEB05513E6A41323EA532843EEB05513E6A41323EA532843EEB05513E6A41323EA532843EEB05513E6A41323EA532843EEB05513E6A41323E"> : tensor<15xf32>
    %16, %17 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x15xui32>)
    %18 = stablehlo.constant dense<9> : tensor<128x15xui32>
    %19 = stablehlo.shift_right_logical %17, %18 : tensor<128x15xui32>
    %20 = stablehlo.convert %19 : (tensor<128x15xui32>) -> tensor<128x15xf32>
    %21 = stablehlo.constant dense<1.1920929E-7> : tensor<128x15xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128x15xf32>
    %23 = stablehlo.constant dense<2.0> : tensor<128x15xf32>
    %24 = stablehlo.constant dense<1.0> : tensor<128x15xf32>
    %25 = stablehlo.multiply %22, %23 : tensor<128x15xf32>
    %26 = stablehlo.subtract %25, %24 : tensor<128x15xf32>
    %27 = chlo.erf_inv %26 : tensor<128x15xf32> -> tensor<128x15xf32>
    %28 = stablehlo.constant dense<1.4142135623730951> : tensor<128x15xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128x15xf32>
    %30, %31 = stablehlo.rng_bit_generator %16, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x15xui32>)
    %32 = stablehlo.constant dense<9> : tensor<128x15xui32>
    %33 = stablehlo.shift_right_logical %31, %32 : tensor<128x15xui32>
    %34 = stablehlo.convert %33 : (tensor<128x15xui32>) -> tensor<128x15xf32>
    %35 = stablehlo.constant dense<1.1920929E-7> : tensor<128x15xf32>
    %36 = stablehlo.multiply %34, %35 : tensor<128x15xf32>
    %37 = stablehlo.constant dense<0> : tensor<i32>
    %38 = stablehlo.constant dense<false> : tensor<15xi1>
    %39 = stablehlo.constant dense<0.0> : tensor<15xf32>
    %43:3 = stablehlo.while(%40 = %37, %41 = %38, %42 = %39) : tensor<i32>, tensor<15xi1>, tensor<15xf32>
    cond {
      %44 = stablehlo.constant dense<128> : tensor<i32>
      %45 = stablehlo.compare LT, %40, %44, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %46 = stablehlo.constant dense<true> : tensor<i1>
      %47 = stablehlo.reduce(%41 init: %46) applies stablehlo.and across dimensions = [0] : (tensor<15xi1>, tensor<i1>) -> tensor<i1>
      %48 = stablehlo.not %47 : tensor<i1>
      %49 = stablehlo.and %45, %48 : tensor<i1>
      stablehlo.return %49 : tensor<i1>
    } do {
      %50 = stablehlo.constant dense<0> : tensor<i32>
      %51 = stablehlo.dynamic_slice %29, %40, %50, sizes = [1, 15] : (tensor<128x15xf32>, tensor<i32>, tensor<i32>) -> tensor<1x15xf32>
      %52 = stablehlo.reshape %51 : (tensor<1x15xf32>) -> tensor<15xf32>
      %53 = stablehlo.dynamic_slice %36, %40, %50, sizes = [1, 15] : (tensor<128x15xf32>, tensor<i32>, tensor<i32>) -> tensor<1x15xf32>
      %54 = stablehlo.reshape %53 : (tensor<1x15xf32>) -> tensor<15xf32>
      %55 = stablehlo.multiply %15, %52 : tensor<15xf32>
      %56 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<15xf32>
      %57 = stablehlo.add %56, %55 : tensor<15xf32>
      %58 = stablehlo.multiply %57, %57 : tensor<15xf32>
      %59 = stablehlo.multiply %58, %57 : tensor<15xf32>
      %60 = stablehlo.multiply %10, %59 : tensor<15xf32>
      %61 = stablehlo.constant dense<0.5> : tensor<f32>
      %62 = stablehlo.multiply %52, %52 : tensor<15xf32>
      %63 = stablehlo.broadcast_in_dim %61, dims = [] : (tensor<f32>) -> tensor<15xf32>
      %64 = stablehlo.multiply %63, %62 : tensor<15xf32>
      %65 = stablehlo.negate %60 : tensor<15xf32>
      %66 = stablehlo.log %59 : tensor<15xf32>
      %67 = stablehlo.multiply %10, %66 : tensor<15xf32>
      %68 = stablehlo.add %64, %10 : tensor<15xf32>
      %69 = stablehlo.add %68, %65 : tensor<15xf32>
      %70 = stablehlo.add %69, %67 : tensor<15xf32>
      %71 = stablehlo.log %54 : tensor<15xf32>
      %72 = stablehlo.compare LT, %71, %70 : (tensor<15xf32>, tensor<15xf32>) -> tensor<15xi1>
      %73 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<15xf32>
      %74 = stablehlo.compare GT, %59, %73 : (tensor<15xf32>, tensor<15xf32>) -> tensor<15xi1>
      %75 = stablehlo.and %72, %74 : tensor<15xi1>
      %76 = stablehlo.select %41, %42, %60 : (tensor<15xi1>, tensor<15xf32>, tensor<15xf32>) -> tensor<15xf32>
      %77 = stablehlo.or %41, %75 : tensor<15xi1>
      %78 = stablehlo.constant dense<1> : tensor<i32>
      %79 = stablehlo.add %40, %78 : tensor<i32>
      stablehlo.return %79, %77, %76 : tensor<i32>, tensor<15xi1>, tensor<15xf32>
    }
    %80, %81 = stablehlo.rng_bit_generator %30, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<15xui32>)
    %82 = stablehlo.constant dense<9> : tensor<15xui32>
    %83 = stablehlo.shift_right_logical %81, %82 : tensor<15xui32>
    %84 = stablehlo.convert %83 : (tensor<15xui32>) -> tensor<15xf32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<15xf32>
    %86 = stablehlo.multiply %84, %85 : tensor<15xf32>
    %87 = stablehlo.log %43#2 : tensor<15xf32>
    %90 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<15xf32>
    %91 = stablehlo.add %87, %90 : tensor<15xf32>
    %92 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [] : (tensor<f32>) -> tensor<15xf32>
    %94 = stablehlo.subtract %91, %93 : tensor<15xf32>
    %95 = stablehlo.reshape %94 : (tensor<15xf32>) -> tensor<5x3xf32>
    %96 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %97 = stablehlo.reduce(%95 init: %96) applies stablehlo.maximum across dimensions = [1] : (tensor<5x3xf32>, tensor<f32>) -> tensor<5xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>
    %99 = stablehlo.subtract %95, %98 : tensor<5x3xf32>
    %100 = stablehlo.exponential %99 : tensor<5x3xf32>
    %101 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %102 = stablehlo.reduce(%100 init: %101) applies stablehlo.add across dimensions = [1] : (tensor<5x3xf32>, tensor<f32>) -> tensor<5xf32>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>
    %104 = stablehlo.divide %100, %103 : tensor<5x3xf32>
    %105 = stablehlo.reshape %104 : (tensor<5x3xf32>) -> tensor<5x3xf32>
    return %105, %80 : tensor<5x3xf32>, tensor<2xui64>
  }
}
