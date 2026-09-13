module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<5.0> : tensor<f32>
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.compare LT, %0, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.constant dense<6.0> : tensor<f32>
    %6 = stablehlo.select %4, %5, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %8 = stablehlo.subtract %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<9.0> : tensor<f32>
    %10 = stablehlo.multiply %9, %8 : tensor<f32>
    %11 = stablehlo.sqrt %10 : tensor<f32>
    %12 = stablehlo.divide %3, %11 : tensor<f32>
    %13, %14 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %15 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %16 = stablehlo.shift_right_logical %14, %15 : tensor<128x4xui32>
    %17 = stablehlo.convert %16 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %18 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x4xf32>
    %20 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %21 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %22 = stablehlo.multiply %19, %20 : tensor<128x4xf32>
    %23 = stablehlo.subtract %22, %21 : tensor<128x4xf32>
    %24 = chlo.erf_inv %23 : tensor<128x4xf32> -> tensor<128x4xf32>
    %25 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<128x4xf32>
    %27, %28 = stablehlo.rng_bit_generator %13, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %29 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %30 = stablehlo.shift_right_logical %28, %29 : tensor<128x4xui32>
    %31 = stablehlo.convert %30 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %32 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128x4xf32>
    %34 = stablehlo.constant dense<0> : tensor<i32>
    %35 = stablehlo.constant dense<false> : tensor<4xi1>
    %36 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %40:3 = stablehlo.while(%37 = %34, %38 = %35, %39 = %36) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %41 = stablehlo.constant dense<128> : tensor<i32>
      %42 = stablehlo.compare LT, %37, %41, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %43 = stablehlo.constant dense<true> : tensor<i1>
      %44 = stablehlo.reduce(%38 init: %43) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %45 = stablehlo.not %44 : tensor<i1>
      %46 = stablehlo.and %42, %45 : tensor<i1>
      stablehlo.return %46 : tensor<i1>
    } do {
      %47 = stablehlo.constant dense<0> : tensor<i32>
      %48 = stablehlo.dynamic_slice %26, %37, %47, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %49 = stablehlo.reshape %48 : (tensor<1x4xf32>) -> tensor<4xf32>
      %50 = stablehlo.dynamic_slice %33, %37, %47, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %51 = stablehlo.reshape %50 : (tensor<1x4xf32>) -> tensor<4xf32>
      %52 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %53 = stablehlo.multiply %52, %49 : tensor<4xf32>
      %54 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %55 = stablehlo.add %54, %53 : tensor<4xf32>
      %56 = stablehlo.multiply %55, %55 : tensor<4xf32>
      %57 = stablehlo.multiply %56, %55 : tensor<4xf32>
      %58 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %59 = stablehlo.multiply %58, %57 : tensor<4xf32>
      %60 = stablehlo.constant dense<0.5> : tensor<f32>
      %61 = stablehlo.multiply %49, %49 : tensor<4xf32>
      %62 = stablehlo.broadcast_in_dim %60, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %63 = stablehlo.multiply %62, %61 : tensor<4xf32>
      %64 = stablehlo.negate %59 : tensor<4xf32>
      %65 = stablehlo.log %57 : tensor<4xf32>
      %66 = stablehlo.multiply %58, %65 : tensor<4xf32>
      %67 = stablehlo.add %63, %58 : tensor<4xf32>
      %68 = stablehlo.add %67, %64 : tensor<4xf32>
      %69 = stablehlo.add %68, %66 : tensor<4xf32>
      %70 = stablehlo.log %51 : tensor<4xf32>
      %71 = stablehlo.compare LT, %70, %69 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %72 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %73 = stablehlo.compare GT, %57, %72 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %74 = stablehlo.and %71, %73 : tensor<4xi1>
      %75 = stablehlo.select %38, %39, %59 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %76 = stablehlo.or %38, %74 : tensor<4xi1>
      %77 = stablehlo.constant dense<1> : tensor<i32>
      %78 = stablehlo.add %37, %77 : tensor<i32>
      stablehlo.return %78, %76, %75 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %79, %80 = stablehlo.rng_bit_generator %27, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %81 = stablehlo.constant dense<9> : tensor<4xui32>
    %82 = stablehlo.shift_right_logical %80, %81 : tensor<4xui32>
    %83 = stablehlo.convert %82 : (tensor<4xui32>) -> tensor<4xf32>
    %84 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %85 = stablehlo.multiply %83, %84 : tensor<4xf32>
    %86 = stablehlo.constant dense<0.20000000298023224> : tensor<f32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %88 = stablehlo.power %85, %87 : tensor<4xf32>
    %89 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %90 = stablehlo.select %4, %88, %89 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %91 = stablehlo.multiply %40#2, %90 : tensor<4xf32>
    %92 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %93 = stablehlo.divide %91, %92 : tensor<4xf32>
    %94, %95 = stablehlo.rng_bit_generator %79, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %96 = stablehlo.constant dense<9> : tensor<4xui32>
    %97 = stablehlo.shift_right_logical %95, %96 : tensor<4xui32>
    %98 = stablehlo.convert %97 : (tensor<4xui32>) -> tensor<4xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<4xf32>
    %101 = stablehlo.negate %93 : tensor<4xf32>
    %102 = stablehlo.exponential %101 : tensor<4xf32>
    %103 = stablehlo.constant dense<false> : tensor<4xi1>
    %109:5 = stablehlo.while(%104 = %2, %105 = %102, %106 = %102, %107 = %103, %108 = %36) : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %110 = stablehlo.constant dense<256.0> : tensor<f32>
      %111 = stablehlo.compare LT, %104, %110 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %112 = stablehlo.constant dense<true> : tensor<i1>
      %113 = stablehlo.reduce(%107 init: %112) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %114 = stablehlo.not %113 : tensor<i1>
      %115 = stablehlo.and %111, %114 : tensor<i1>
      stablehlo.return %115 : tensor<i1>
    } do {
      %116 = stablehlo.compare LE, %100, %105 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %117 = stablehlo.broadcast_in_dim %104, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %118 = stablehlo.select %107, %108, %117 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %119 = stablehlo.or %107, %116 : tensor<4xi1>
      %120 = stablehlo.constant dense<1.0> : tensor<f32>
      %121 = stablehlo.add %104, %120 : tensor<f32>
      %122 = stablehlo.broadcast_in_dim %121, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %123 = stablehlo.divide %93, %122 : tensor<4xf32>
      %124 = stablehlo.multiply %106, %123 : tensor<4xf32>
      %125 = stablehlo.add %105, %124 : tensor<4xf32>
      stablehlo.return %121, %125, %124, %119, %118 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    }
    return %109#4, %94 : tensor<4xf32>, tensor<2xui64>
  }
}
