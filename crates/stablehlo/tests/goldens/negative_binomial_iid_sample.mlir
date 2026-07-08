module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<5.0> : tensor<f32>
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.compare LT, %0, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.add %0, %3 : tensor<f32>
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
      %50 = stablehlo.constant dense<0> : tensor<i32>
      %51 = stablehlo.dynamic_slice %33, %37, %50, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %52 = stablehlo.reshape %51 : (tensor<1x4xf32>) -> tensor<4xf32>
      %53 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %54 = stablehlo.multiply %53, %49 : tensor<4xf32>
      %55 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %56 = stablehlo.add %55, %54 : tensor<4xf32>
      %57 = stablehlo.multiply %56, %56 : tensor<4xf32>
      %58 = stablehlo.multiply %57, %56 : tensor<4xf32>
      %59 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %60 = stablehlo.multiply %59, %58 : tensor<4xf32>
      %61 = stablehlo.constant dense<0.5> : tensor<f32>
      %62 = stablehlo.multiply %49, %49 : tensor<4xf32>
      %63 = stablehlo.broadcast_in_dim %61, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %64 = stablehlo.multiply %63, %62 : tensor<4xf32>
      %65 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %66 = stablehlo.multiply %65, %58 : tensor<4xf32>
      %67 = stablehlo.negate %66 : tensor<4xf32>
      %68 = stablehlo.log %58 : tensor<4xf32>
      %69 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %70 = stablehlo.multiply %69, %68 : tensor<4xf32>
      %71 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %72 = stablehlo.add %64, %71 : tensor<4xf32>
      %73 = stablehlo.add %72, %67 : tensor<4xf32>
      %74 = stablehlo.add %73, %70 : tensor<4xf32>
      %75 = stablehlo.log %52 : tensor<4xf32>
      %76 = stablehlo.compare LT, %75, %74 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %77 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %78 = stablehlo.compare GT, %58, %77 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %79 = stablehlo.and %76, %78 : tensor<4xi1>
      %80 = stablehlo.select %38, %39, %60 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %81 = stablehlo.or %38, %79 : tensor<4xi1>
      %82 = stablehlo.constant dense<1> : tensor<i32>
      %83 = stablehlo.add %37, %82 : tensor<i32>
      stablehlo.return %83, %81, %80 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %84, %85 = stablehlo.rng_bit_generator %27, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %86 = stablehlo.constant dense<9> : tensor<4xui32>
    %87 = stablehlo.shift_right_logical %85, %86 : tensor<4xui32>
    %88 = stablehlo.convert %87 : (tensor<4xui32>) -> tensor<4xf32>
    %89 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %90 = stablehlo.multiply %88, %89 : tensor<4xf32>
    %91 = stablehlo.divide %3, %0 : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %93 = stablehlo.power %90, %92 : tensor<4xf32>
    %94 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %95 = stablehlo.select %4, %93, %94 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %96 = stablehlo.multiply %40#2, %95 : tensor<4xf32>
    %97 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %98 = stablehlo.divide %96, %97 : tensor<4xf32>
    %99, %100 = stablehlo.rng_bit_generator %84, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %101 = stablehlo.constant dense<9> : tensor<4xui32>
    %102 = stablehlo.shift_right_logical %100, %101 : tensor<4xui32>
    %103 = stablehlo.convert %102 : (tensor<4xui32>) -> tensor<4xf32>
    %104 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %105 = stablehlo.multiply %103, %104 : tensor<4xf32>
    %106 = stablehlo.negate %98 : tensor<4xf32>
    %107 = stablehlo.exponential %106 : tensor<4xf32>
    %108 = stablehlo.constant dense<0.0> : tensor<f32>
    %109 = stablehlo.constant dense<false> : tensor<4xi1>
    %110 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %116:5 = stablehlo.while(%111 = %108, %112 = %107, %113 = %107, %114 = %109, %115 = %110) : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %117 = stablehlo.constant dense<256.0> : tensor<f32>
      %118 = stablehlo.compare LT, %111, %117 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %119 = stablehlo.constant dense<true> : tensor<i1>
      %120 = stablehlo.reduce(%114 init: %119) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %121 = stablehlo.not %120 : tensor<i1>
      %122 = stablehlo.and %118, %121 : tensor<i1>
      stablehlo.return %122 : tensor<i1>
    } do {
      %123 = stablehlo.compare LE, %105, %112 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %124 = stablehlo.broadcast_in_dim %111, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %125 = stablehlo.select %114, %115, %124 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %126 = stablehlo.or %114, %123 : tensor<4xi1>
      %127 = stablehlo.constant dense<1.0> : tensor<f32>
      %128 = stablehlo.add %111, %127 : tensor<f32>
      %129 = stablehlo.broadcast_in_dim %128, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %130 = stablehlo.divide %98, %129 : tensor<4xf32>
      %131 = stablehlo.multiply %113, %130 : tensor<4xf32>
      %132 = stablehlo.add %112, %131 : tensor<4xf32>
      stablehlo.return %128, %132, %131, %126, %125 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    }
    return %116#4, %99 : tensor<4xf32>, tensor<2xui64>
  }
}
