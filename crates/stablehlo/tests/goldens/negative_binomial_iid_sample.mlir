module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %7 = stablehlo.constant dense<4.666666507720947> : tensor<f32>
    %11 = stablehlo.constant dense<0.15430335700511932> : tensor<f32>
    %12, %13 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %14 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %15 = stablehlo.shift_right_logical %13, %14 : tensor<128x4xui32>
    %16 = stablehlo.convert %15 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %17 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128x4xf32>
    %19 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %20 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %21 = stablehlo.multiply %18, %19 : tensor<128x4xf32>
    %22 = stablehlo.subtract %21, %20 : tensor<128x4xf32>
    %23 = chlo.erf_inv %22 : tensor<128x4xf32> -> tensor<128x4xf32>
    %24 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<128x4xf32>
    %26, %27 = stablehlo.rng_bit_generator %12, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %28 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %29 = stablehlo.shift_right_logical %27, %28 : tensor<128x4xui32>
    %30 = stablehlo.convert %29 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %31 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<128x4xf32>
    %33 = stablehlo.constant dense<0> : tensor<i32>
    %34 = stablehlo.constant dense<false> : tensor<4xi1>
    %35 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %39:3 = stablehlo.while(%36 = %33, %37 = %34, %38 = %35) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %40 = stablehlo.constant dense<128> : tensor<i32>
      %41 = stablehlo.compare LT, %36, %40, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %42 = stablehlo.constant dense<true> : tensor<i1>
      %43 = stablehlo.reduce(%37 init: %42) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %44 = stablehlo.not %43 : tensor<i1>
      %45 = stablehlo.and %41, %44 : tensor<i1>
      stablehlo.return %45 : tensor<i1>
    } do {
      %46 = stablehlo.constant dense<0> : tensor<i32>
      %47 = stablehlo.dynamic_slice %25, %36, %46, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %48 = stablehlo.reshape %47 : (tensor<1x4xf32>) -> tensor<4xf32>
      %49 = stablehlo.dynamic_slice %32, %36, %46, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x4xf32>) -> tensor<4xf32>
      %51 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %52 = stablehlo.multiply %51, %48 : tensor<4xf32>
      %53 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %54 = stablehlo.add %53, %52 : tensor<4xf32>
      %55 = stablehlo.multiply %54, %54 : tensor<4xf32>
      %56 = stablehlo.multiply %55, %54 : tensor<4xf32>
      %57 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %58 = stablehlo.multiply %57, %56 : tensor<4xf32>
      %59 = stablehlo.constant dense<0.5> : tensor<f32>
      %60 = stablehlo.multiply %48, %48 : tensor<4xf32>
      %61 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %62 = stablehlo.multiply %61, %60 : tensor<4xf32>
      %63 = stablehlo.negate %58 : tensor<4xf32>
      %64 = stablehlo.log %56 : tensor<4xf32>
      %65 = stablehlo.multiply %57, %64 : tensor<4xf32>
      %66 = stablehlo.add %62, %57 : tensor<4xf32>
      %67 = stablehlo.add %66, %63 : tensor<4xf32>
      %68 = stablehlo.add %67, %65 : tensor<4xf32>
      %69 = stablehlo.log %50 : tensor<4xf32>
      %70 = stablehlo.compare LT, %69, %68 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %71 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %72 = stablehlo.compare GT, %56, %71 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %73 = stablehlo.and %70, %72 : tensor<4xi1>
      %74 = stablehlo.select %37, %38, %58 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %75 = stablehlo.or %37, %73 : tensor<4xi1>
      %76 = stablehlo.constant dense<1> : tensor<i32>
      %77 = stablehlo.add %36, %76 : tensor<i32>
      stablehlo.return %77, %75, %74 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %78, %79 = stablehlo.rng_bit_generator %26, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %80 = stablehlo.constant dense<9> : tensor<4xui32>
    %81 = stablehlo.shift_right_logical %79, %80 : tensor<4xui32>
    %82 = stablehlo.convert %81 : (tensor<4xui32>) -> tensor<4xf32>
    %83 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %84 = stablehlo.multiply %82, %83 : tensor<4xf32>
    %88 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %89 = stablehlo.multiply %39#2, %88 : tensor<4xf32>
    %90 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %91 = stablehlo.divide %89, %90 : tensor<4xf32>
    %92, %93 = stablehlo.rng_bit_generator %78, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %94 = stablehlo.constant dense<9> : tensor<4xui32>
    %95 = stablehlo.shift_right_logical %93, %94 : tensor<4xui32>
    %96 = stablehlo.convert %95 : (tensor<4xui32>) -> tensor<4xf32>
    %97 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %98 = stablehlo.multiply %96, %97 : tensor<4xf32>
    %99 = stablehlo.negate %91 : tensor<4xf32>
    %100 = stablehlo.exponential %99 : tensor<4xf32>
    %101 = stablehlo.constant dense<false> : tensor<4xi1>
    %107:5 = stablehlo.while(%102 = %2, %103 = %100, %104 = %100, %105 = %101, %106 = %35) : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %108 = stablehlo.constant dense<256.0> : tensor<f32>
      %109 = stablehlo.compare LT, %102, %108 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %110 = stablehlo.constant dense<true> : tensor<i1>
      %111 = stablehlo.reduce(%105 init: %110) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %112 = stablehlo.not %111 : tensor<i1>
      %113 = stablehlo.and %109, %112 : tensor<i1>
      stablehlo.return %113 : tensor<i1>
    } do {
      %114 = stablehlo.compare LE, %98, %103 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %115 = stablehlo.broadcast_in_dim %102, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %116 = stablehlo.select %105, %106, %115 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %117 = stablehlo.or %105, %114 : tensor<4xi1>
      %118 = stablehlo.constant dense<1.0> : tensor<f32>
      %119 = stablehlo.add %102, %118 : tensor<f32>
      %120 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %121 = stablehlo.divide %91, %120 : tensor<4xf32>
      %122 = stablehlo.multiply %104, %121 : tensor<4xf32>
      %123 = stablehlo.add %103, %122 : tensor<4xf32>
      stablehlo.return %119, %123, %122, %117, %116 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    }
    return %107#4, %92 : tensor<4xf32>, tensor<2xui64>
  }
}
