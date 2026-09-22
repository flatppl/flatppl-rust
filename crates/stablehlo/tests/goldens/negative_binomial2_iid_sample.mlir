module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<5.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.6666666269302368> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %1, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.constant dense<6.0> : tensor<f32>
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
      %51 = stablehlo.dynamic_slice %34, %38, %48, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %52 = stablehlo.reshape %51 : (tensor<1x4xf32>) -> tensor<4xf32>
      %53 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %54 = stablehlo.multiply %53, %50 : tensor<4xf32>
      %55 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %56 = stablehlo.add %55, %54 : tensor<4xf32>
      %57 = stablehlo.multiply %56, %56 : tensor<4xf32>
      %58 = stablehlo.multiply %57, %56 : tensor<4xf32>
      %59 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %60 = stablehlo.multiply %59, %58 : tensor<4xf32>
      %61 = stablehlo.constant dense<0.5> : tensor<f32>
      %62 = stablehlo.multiply %50, %50 : tensor<4xf32>
      %63 = stablehlo.broadcast_in_dim %61, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %64 = stablehlo.multiply %63, %62 : tensor<4xf32>
      %65 = stablehlo.negate %60 : tensor<4xf32>
      %66 = stablehlo.log %58 : tensor<4xf32>
      %67 = stablehlo.multiply %59, %66 : tensor<4xf32>
      %68 = stablehlo.add %64, %59 : tensor<4xf32>
      %69 = stablehlo.add %68, %65 : tensor<4xf32>
      %70 = stablehlo.add %69, %67 : tensor<4xf32>
      %71 = stablehlo.log %52 : tensor<4xf32>
      %72 = stablehlo.compare LT, %71, %70 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %73 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %74 = stablehlo.compare GT, %58, %73 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %75 = stablehlo.and %72, %74 : tensor<4xi1>
      %76 = stablehlo.select %39, %40, %60 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %77 = stablehlo.or %39, %75 : tensor<4xi1>
      %78 = stablehlo.constant dense<1> : tensor<i32>
      %79 = stablehlo.add %38, %78 : tensor<i32>
      stablehlo.return %79, %77, %76 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %80, %81 = stablehlo.rng_bit_generator %28, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %82 = stablehlo.constant dense<9> : tensor<4xui32>
    %83 = stablehlo.shift_right_logical %81, %82 : tensor<4xui32>
    %84 = stablehlo.convert %83 : (tensor<4xui32>) -> tensor<4xf32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %86 = stablehlo.multiply %84, %85 : tensor<4xf32>
    %87 = stablehlo.constant dense<0.20000000298023224> : tensor<f32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %89 = stablehlo.power %86, %88 : tensor<4xf32>
    %90 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %91 = stablehlo.select %5, %89, %90 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %92 = stablehlo.multiply %41#2, %91 : tensor<4xf32>
    %93 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %94 = stablehlo.divide %92, %93 : tensor<4xf32>
    %95, %96 = stablehlo.rng_bit_generator %80, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %97 = stablehlo.constant dense<9> : tensor<4xui32>
    %98 = stablehlo.shift_right_logical %96, %97 : tensor<4xui32>
    %99 = stablehlo.convert %98 : (tensor<4xui32>) -> tensor<4xf32>
    %100 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %101 = stablehlo.multiply %99, %100 : tensor<4xf32>
    %102 = stablehlo.negate %94 : tensor<4xf32>
    %103 = stablehlo.exponential %102 : tensor<4xf32>
    %104 = stablehlo.constant dense<false> : tensor<4xi1>
    %110:5 = stablehlo.while(%105 = %3, %106 = %103, %107 = %103, %108 = %104, %109 = %37) : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %111 = stablehlo.constant dense<256.0> : tensor<f32>
      %112 = stablehlo.compare LT, %105, %111 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %113 = stablehlo.constant dense<true> : tensor<i1>
      %114 = stablehlo.reduce(%108 init: %113) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %115 = stablehlo.not %114 : tensor<i1>
      %116 = stablehlo.and %112, %115 : tensor<i1>
      stablehlo.return %116 : tensor<i1>
    } do {
      %117 = stablehlo.compare LE, %101, %106 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %118 = stablehlo.broadcast_in_dim %105, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %119 = stablehlo.select %108, %109, %118 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %120 = stablehlo.or %108, %117 : tensor<4xi1>
      %121 = stablehlo.constant dense<1.0> : tensor<f32>
      %122 = stablehlo.add %105, %121 : tensor<f32>
      %123 = stablehlo.broadcast_in_dim %122, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %124 = stablehlo.divide %94, %123 : tensor<4xf32>
      %125 = stablehlo.multiply %107, %124 : tensor<4xf32>
      %126 = stablehlo.add %106, %125 : tensor<4xf32>
      stablehlo.return %122, %126, %125, %120, %119 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xi1>, tensor<4xf32>
    }
    return %110#4, %95 : tensor<4xf32>, tensor<2xui64>
  }
}
