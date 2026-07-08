module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
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
    %14, %15 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %16 = stablehlo.constant dense<9> : tensor<128xui32>
    %17 = stablehlo.shift_right_logical %15, %16 : tensor<128xui32>
    %18 = stablehlo.convert %17 : (tensor<128xui32>) -> tensor<128xf32>
    %19 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128xf32>
    %21 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %22 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %23 = stablehlo.multiply %20, %21 : tensor<128xf32>
    %24 = stablehlo.subtract %23, %22 : tensor<128xf32>
    %25 = chlo.erf_inv %24 : tensor<128xf32> -> tensor<128xf32>
    %26 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128xf32>
    %28 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %29 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %30 = stablehlo.multiply %27, %28 : tensor<128xf32>
    %31 = stablehlo.add %30, %29 : tensor<128xf32>
    %32, %33 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %34 = stablehlo.constant dense<9> : tensor<128xui32>
    %35 = stablehlo.shift_right_logical %33, %34 : tensor<128xui32>
    %36 = stablehlo.convert %35 : (tensor<128xui32>) -> tensor<128xf32>
    %37 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %38 = stablehlo.multiply %36, %37 : tensor<128xf32>
    %39 = stablehlo.subtract %4, %3 : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %41 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %42 = stablehlo.multiply %38, %40 : tensor<128xf32>
    %43 = stablehlo.add %42, %41 : tensor<128xf32>
    %44 = stablehlo.constant dense<0> : tensor<i32>
    %45 = stablehlo.constant dense<false> : tensor<i1>
    %46 = stablehlo.constant dense<0.0> : tensor<f32>
    %50:3 = stablehlo.while(%47 = %44, %48 = %45, %49 = %46) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %51 = stablehlo.constant dense<128> : tensor<i32>
      %52 = stablehlo.compare LT, %47, %51, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %53 = stablehlo.not %48 : tensor<i1>
      %54 = stablehlo.and %53, %52 : tensor<i1>
      stablehlo.return %54 : tensor<i1>
    } do {
      %55 = stablehlo.dynamic_slice %31, %47, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %56 = stablehlo.reshape %55 : (tensor<1xf32>) -> tensor<f32>
      %57 = stablehlo.dynamic_slice %43, %47, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %58 = stablehlo.reshape %57 : (tensor<1xf32>) -> tensor<f32>
      %59 = stablehlo.multiply %13, %56 : tensor<f32>
      %60 = stablehlo.add %4, %59 : tensor<f32>
      %61 = stablehlo.multiply %60, %60 : tensor<f32>
      %62 = stablehlo.multiply %61, %60 : tensor<f32>
      %63 = stablehlo.multiply %9, %62 : tensor<f32>
      %64 = stablehlo.constant dense<0.5> : tensor<f32>
      %65 = stablehlo.multiply %56, %56 : tensor<f32>
      %66 = stablehlo.multiply %64, %65 : tensor<f32>
      %67 = stablehlo.multiply %9, %62 : tensor<f32>
      %68 = stablehlo.negate %67 : tensor<f32>
      %69 = stablehlo.log %62 : tensor<f32>
      %70 = stablehlo.multiply %9, %69 : tensor<f32>
      %71 = stablehlo.add %66, %9 : tensor<f32>
      %72 = stablehlo.add %71, %68 : tensor<f32>
      %73 = stablehlo.add %72, %70 : tensor<f32>
      %74 = stablehlo.log %58 : tensor<f32>
      %75 = stablehlo.compare LT, %74, %73 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %76 = stablehlo.compare GT, %62, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %77 = stablehlo.and %75, %76 : tensor<i1>
      %78 = stablehlo.constant dense<1> : tensor<i32>
      %79 = stablehlo.add %47, %78 : tensor<i32>
      stablehlo.return %79, %77, %63 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %80, %81 = stablehlo.rng_bit_generator %32, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %82 = stablehlo.constant dense<9> : tensor<ui32>
    %83 = stablehlo.shift_right_logical %81, %82 : tensor<ui32>
    %84 = stablehlo.convert %83 : (tensor<ui32>) -> tensor<f32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %86 = stablehlo.multiply %84, %85 : tensor<f32>
    %87 = stablehlo.subtract %4, %3 : tensor<f32>
    %88 = stablehlo.multiply %86, %87 : tensor<f32>
    %89 = stablehlo.add %88, %3 : tensor<f32>
    %90 = stablehlo.divide %4, %1 : tensor<f32>
    %91 = stablehlo.power %89, %90 : tensor<f32>
    %92 = stablehlo.select %5, %91, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %93 = stablehlo.multiply %50#2, %92 : tensor<f32>
    %94 = stablehlo.divide %93, %2 : tensor<f32>
    %95 = stablehlo.constant dense<0.0> : tensor<f32>
    %96 = stablehlo.constant dense<1.0> : tensor<f32>
    %97, %98 = stablehlo.rng_bit_generator %80, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %99 = stablehlo.constant dense<9> : tensor<ui32>
    %100 = stablehlo.shift_right_logical %98, %99 : tensor<ui32>
    %101 = stablehlo.convert %100 : (tensor<ui32>) -> tensor<f32>
    %102 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %103 = stablehlo.multiply %101, %102 : tensor<f32>
    %104 = stablehlo.subtract %96, %95 : tensor<f32>
    %105 = stablehlo.multiply %103, %104 : tensor<f32>
    %106 = stablehlo.add %105, %95 : tensor<f32>
    %107 = stablehlo.negate %94 : tensor<f32>
    %108 = stablehlo.exponential %107 : tensor<f32>
    %109 = stablehlo.constant dense<0.0> : tensor<f32>
    %110 = stablehlo.constant dense<false> : tensor<i1>
    %111 = stablehlo.constant dense<0.0> : tensor<f32>
    %117:5 = stablehlo.while(%112 = %109, %113 = %108, %114 = %108, %115 = %110, %116 = %111) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %118 = stablehlo.constant dense<256.0> : tensor<f32>
      %119 = stablehlo.compare LT, %112, %118 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %120 = stablehlo.not %115 : tensor<i1>
      %121 = stablehlo.and %120, %119 : tensor<i1>
      stablehlo.return %121 : tensor<i1>
    } do {
      %122 = stablehlo.compare LE, %106, %113 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %123 = stablehlo.constant dense<1.0> : tensor<f32>
      %124 = stablehlo.add %112, %123 : tensor<f32>
      %125 = stablehlo.divide %94, %124 : tensor<f32>
      %126 = stablehlo.multiply %114, %125 : tensor<f32>
      %127 = stablehlo.add %113, %126 : tensor<f32>
      stablehlo.return %124, %127, %126, %122, %112 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %117#4, %97 : tensor<f32>, tensor<2xui64>
  }
}
