module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %2 = stablehlo.constant dense<1.6666666269302368> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.constant dense<false> : tensor<i1>
    %8 = stablehlo.constant dense<4.666666507720947> : tensor<f32>
    %12 = stablehlo.constant dense<0.15430335700511932> : tensor<f32>
    %13, %14 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %15 = stablehlo.constant dense<9> : tensor<128xui32>
    %16 = stablehlo.shift_right_logical %14, %15 : tensor<128xui32>
    %17 = stablehlo.convert %16 : (tensor<128xui32>) -> tensor<128xf32>
    %18 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128xf32>
    %20 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %21 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %22 = stablehlo.multiply %19, %20 : tensor<128xf32>
    %23 = stablehlo.subtract %22, %21 : tensor<128xf32>
    %24 = chlo.erf_inv %23 : tensor<128xf32> -> tensor<128xf32>
    %25 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<128xf32>
    %27, %28 = stablehlo.rng_bit_generator %13, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %29 = stablehlo.constant dense<9> : tensor<128xui32>
    %30 = stablehlo.shift_right_logical %28, %29 : tensor<128xui32>
    %31 = stablehlo.convert %30 : (tensor<128xui32>) -> tensor<128xf32>
    %32 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128xf32>
    %34 = stablehlo.constant dense<0> : tensor<i32>
    %38:3 = stablehlo.while(%35 = %34, %36 = %5, %37 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %39 = stablehlo.constant dense<128> : tensor<i32>
      %40 = stablehlo.compare LT, %35, %39, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %41 = stablehlo.not %36 : tensor<i1>
      %42 = stablehlo.and %41, %40 : tensor<i1>
      stablehlo.return %42 : tensor<i1>
    } do {
      %43 = stablehlo.dynamic_slice %26, %35, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %44 = stablehlo.reshape %43 : (tensor<1xf32>) -> tensor<f32>
      %45 = stablehlo.dynamic_slice %33, %35, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %46 = stablehlo.reshape %45 : (tensor<1xf32>) -> tensor<f32>
      %47 = stablehlo.multiply %12, %44 : tensor<f32>
      %48 = stablehlo.add %4, %47 : tensor<f32>
      %49 = stablehlo.multiply %48, %48 : tensor<f32>
      %50 = stablehlo.multiply %49, %48 : tensor<f32>
      %51 = stablehlo.multiply %8, %50 : tensor<f32>
      %52 = stablehlo.constant dense<0.5> : tensor<f32>
      %53 = stablehlo.multiply %44, %44 : tensor<f32>
      %54 = stablehlo.multiply %52, %53 : tensor<f32>
      %55 = stablehlo.negate %51 : tensor<f32>
      %56 = stablehlo.log %50 : tensor<f32>
      %57 = stablehlo.multiply %8, %56 : tensor<f32>
      %58 = stablehlo.add %54, %8 : tensor<f32>
      %59 = stablehlo.add %58, %55 : tensor<f32>
      %60 = stablehlo.add %59, %57 : tensor<f32>
      %61 = stablehlo.log %46 : tensor<f32>
      %62 = stablehlo.compare LT, %61, %60 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %63 = stablehlo.compare GT, %50, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %64 = stablehlo.and %62, %63 : tensor<i1>
      %65 = stablehlo.constant dense<1> : tensor<i32>
      %66 = stablehlo.add %35, %65 : tensor<i32>
      stablehlo.return %66, %64, %51 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %67, %68 = stablehlo.rng_bit_generator %27, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %69 = stablehlo.constant dense<9> : tensor<ui32>
    %70 = stablehlo.shift_right_logical %68, %69 : tensor<ui32>
    %71 = stablehlo.convert %70 : (tensor<ui32>) -> tensor<f32>
    %72 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %73 = stablehlo.multiply %71, %72 : tensor<f32>
    %76 = stablehlo.multiply %38#2, %4 : tensor<f32>
    %77 = stablehlo.divide %76, %2 : tensor<f32>
    %78, %79 = stablehlo.rng_bit_generator %67, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %80 = stablehlo.constant dense<9> : tensor<ui32>
    %81 = stablehlo.shift_right_logical %79, %80 : tensor<ui32>
    %82 = stablehlo.convert %81 : (tensor<ui32>) -> tensor<f32>
    %83 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %84 = stablehlo.multiply %82, %83 : tensor<f32>
    %85 = stablehlo.negate %77 : tensor<f32>
    %86 = stablehlo.exponential %85 : tensor<f32>
    %92:5 = stablehlo.while(%87 = %3, %88 = %86, %89 = %86, %90 = %5, %91 = %3) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %93 = stablehlo.constant dense<256.0> : tensor<f32>
      %94 = stablehlo.compare LT, %87, %93 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %95 = stablehlo.not %90 : tensor<i1>
      %96 = stablehlo.and %95, %94 : tensor<i1>
      stablehlo.return %96 : tensor<i1>
    } do {
      %97 = stablehlo.compare LE, %84, %88 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %98 = stablehlo.constant dense<1.0> : tensor<f32>
      %99 = stablehlo.add %87, %98 : tensor<f32>
      %100 = stablehlo.divide %77, %99 : tensor<f32>
      %101 = stablehlo.multiply %89, %100 : tensor<f32>
      %102 = stablehlo.add %88, %101 : tensor<f32>
      stablehlo.return %99, %102, %101, %97, %87 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %92#4, %78 : tensor<f32>, tensor<2xui64>
  }
}
