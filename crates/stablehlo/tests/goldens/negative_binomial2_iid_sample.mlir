module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.constant dense<5.0> : tensor<f32>
    %2 = stablehlo.divide %1, %0 : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %1, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.add %1, %4 : tensor<f32>
    %7 = stablehlo.select %5, %6, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %4, %12 : tensor<f32>
    %14, %15 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %16 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %17 = stablehlo.shift_right_logical %15, %16 : tensor<128x4xui32>
    %18 = stablehlo.convert %17 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %19 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x4xf32>
    %21 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %22 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %23 = stablehlo.multiply %20, %21 : tensor<128x4xf32>
    %24 = stablehlo.subtract %23, %22 : tensor<128x4xf32>
    %25 = chlo.erf_inv %24 : tensor<128x4xf32> -> tensor<128x4xf32>
    %26 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x4xf32>
    %28, %29 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %30 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %31 = stablehlo.shift_right_logical %29, %30 : tensor<128x4xui32>
    %32 = stablehlo.convert %31 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %33 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<128x4xf32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.constant dense<false> : tensor<4xi1>
    %37 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %41:3 = stablehlo.while(%38 = %35, %39 = %36, %40 = %37) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %42 = stablehlo.constant dense<128> : tensor<i32>
      %43 = stablehlo.compare LT, %38, %42, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %44 = stablehlo.constant dense<true> : tensor<i1>
      %45 = stablehlo.reduce(%39 init: %44) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %46 = stablehlo.not %45 : tensor<i1>
      %47 = stablehlo.and %43, %46 : tensor<i1>
      stablehlo.return %47 : tensor<i1>
    } do {
      %48 = stablehlo.constant dense<0> : tensor<i32>
      %49 = stablehlo.dynamic_slice %27, %38, %48, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x4xf32>) -> tensor<4xf32>
      %51 = stablehlo.constant dense<0> : tensor<i32>
      %52 = stablehlo.dynamic_slice %34, %38, %51, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %53 = stablehlo.reshape %52 : (tensor<1x4xf32>) -> tensor<4xf32>
      %54 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %55 = stablehlo.multiply %54, %50 : tensor<4xf32>
      %56 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %57 = stablehlo.add %56, %55 : tensor<4xf32>
      %58 = stablehlo.multiply %57, %57 : tensor<4xf32>
      %59 = stablehlo.multiply %58, %57 : tensor<4xf32>
      %60 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %61 = stablehlo.multiply %60, %59 : tensor<4xf32>
      %62 = stablehlo.constant dense<0.5> : tensor<f32>
      %63 = stablehlo.multiply %50, %50 : tensor<4xf32>
      %64 = stablehlo.broadcast_in_dim %62, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %65 = stablehlo.multiply %64, %63 : tensor<4xf32>
      %66 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %67 = stablehlo.multiply %66, %59 : tensor<4xf32>
      %68 = stablehlo.negate %67 : tensor<4xf32>
      %69 = stablehlo.log %59 : tensor<4xf32>
      %70 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %71 = stablehlo.multiply %70, %69 : tensor<4xf32>
      %72 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %73 = stablehlo.add %65, %72 : tensor<4xf32>
      %74 = stablehlo.add %73, %68 : tensor<4xf32>
      %75 = stablehlo.add %74, %71 : tensor<4xf32>
      %76 = stablehlo.log %53 : tensor<4xf32>
      %77 = stablehlo.compare LT, %76, %75 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %78 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %79 = stablehlo.compare GT, %59, %78 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %80 = stablehlo.and %77, %79 : tensor<4xi1>
      %81 = stablehlo.select %39, %40, %61 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %82 = stablehlo.or %39, %80 : tensor<4xi1>
      %83 = stablehlo.constant dense<1> : tensor<i32>
      %84 = stablehlo.add %38, %83 : tensor<i32>
      stablehlo.return %84, %82, %81 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %85, %86 = stablehlo.rng_bit_generator %28, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %87 = stablehlo.constant dense<9> : tensor<4xui32>
    %88 = stablehlo.shift_right_logical %86, %87 : tensor<4xui32>
    %89 = stablehlo.convert %88 : (tensor<4xui32>) -> tensor<4xf32>
    %90 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %91 = stablehlo.multiply %89, %90 : tensor<4xf32>
    %92 = stablehlo.divide %4, %1 : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %94 = stablehlo.power %91, %93 : tensor<4xf32>
    %95 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %96 = stablehlo.select %5, %94, %95 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %97 = stablehlo.multiply %41#2, %96 : tensor<4xf32>
    %98 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %99 = stablehlo.divide %97, %98 : tensor<4xf32>
    %100, %101 = stablehlo.rng_bit_generator %85, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %102 = stablehlo.constant dense<9> : tensor<4xui32>
    %103 = stablehlo.shift_right_logical %101, %102 : tensor<4xui32>
    %104 = stablehlo.convert %103 : (tensor<4xui32>) -> tensor<4xf32>
    %105 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %106 = stablehlo.multiply %104, %105 : tensor<4xf32>
    %107 = stablehlo.negate %99 : tensor<4xf32>
    %108 = stablehlo.exponential %107 : tensor<4xf32>
    %109 = stablehlo.constant dense<0.0> : tensor<f32>
    %110 = stablehlo.constant dense<false> : tensor<4xi1>
    %111 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %117:5 = stablehlo.while(%112 = %109, %113 = %108, %114 = %108, %115 = %110, %116 = %111) : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %118 = stablehlo.constant dense<256.0> : tensor<f32>
      %119 = stablehlo.compare LT, %112, %118 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %120 = stablehlo.constant dense<true> : tensor<i1>
      %121 = stablehlo.reduce(%115 init: %120) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %122 = stablehlo.not %121 : tensor<i1>
      %123 = stablehlo.and %119, %122 : tensor<i1>
      stablehlo.return %123 : tensor<i1>
    } do {
      %124 = stablehlo.compare LE, %106, %113 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %125 = stablehlo.broadcast_in_dim %112, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %126 = stablehlo.select %115, %116, %125 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %127 = stablehlo.or %115, %124 : tensor<4xi1>
      %128 = stablehlo.constant dense<1.0> : tensor<f32>
      %129 = stablehlo.add %112, %128 : tensor<f32>
      %130 = stablehlo.broadcast_in_dim %129, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %131 = stablehlo.divide %99, %130 : tensor<4xf32>
      %132 = stablehlo.multiply %114, %131 : tensor<4xf32>
      %133 = stablehlo.add %113, %132 : tensor<4xf32>
      stablehlo.return %129, %133, %132, %127, %126 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    }
    return %117#4, %100 : tensor<4xf32>, tensor<2xui64>
  }
}
