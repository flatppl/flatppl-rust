module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.constant dense<false> : tensor<i1>
    %7 = stablehlo.constant dense<"0x55559540"> : tensor<f32>
    %11 = stablehlo.constant dense<"0xB3011E3E"> : tensor<f32>
    %12, %13 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %14 = stablehlo.constant dense<9> : tensor<128xui32>
    %15 = stablehlo.shift_right_logical %13, %14 : tensor<128xui32>
    %16 = stablehlo.convert %15 : (tensor<128xui32>) -> tensor<128xf32>
    %17 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128xf32>
    %19 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %20 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %21 = stablehlo.multiply %18, %19 : tensor<128xf32>
    %22 = stablehlo.subtract %21, %20 : tensor<128xf32>
    %23 = chlo.erf_inv %22 : tensor<128xf32> -> tensor<128xf32>
    %24 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<128xf32>
    %26, %27 = stablehlo.rng_bit_generator %12, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %28 = stablehlo.constant dense<9> : tensor<128xui32>
    %29 = stablehlo.shift_right_logical %27, %28 : tensor<128xui32>
    %30 = stablehlo.convert %29 : (tensor<128xui32>) -> tensor<128xf32>
    %31 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<128xf32>
    %33 = stablehlo.constant dense<0> : tensor<i32>
    %37:3 = stablehlo.while(%34 = %33, %35 = %4, %36 = %2) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %38 = stablehlo.constant dense<128> : tensor<i32>
      %39 = stablehlo.compare LT, %34, %38, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %40 = stablehlo.not %35 : tensor<i1>
      %41 = stablehlo.and %40, %39 : tensor<i1>
      stablehlo.return %41 : tensor<i1>
    } do {
      %42 = stablehlo.dynamic_slice %25, %34, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %43 = stablehlo.reshape %42 : (tensor<1xf32>) -> tensor<f32>
      %44 = stablehlo.dynamic_slice %32, %34, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %45 = stablehlo.reshape %44 : (tensor<1xf32>) -> tensor<f32>
      %46 = stablehlo.multiply %11, %43 : tensor<f32>
      %47 = stablehlo.add %3, %46 : tensor<f32>
      %48 = stablehlo.multiply %47, %47 : tensor<f32>
      %49 = stablehlo.multiply %48, %47 : tensor<f32>
      %50 = stablehlo.multiply %7, %49 : tensor<f32>
      %51 = stablehlo.constant dense<0.5> : tensor<f32>
      %52 = stablehlo.multiply %43, %43 : tensor<f32>
      %53 = stablehlo.multiply %51, %52 : tensor<f32>
      %54 = stablehlo.negate %50 : tensor<f32>
      %55 = stablehlo.log %49 : tensor<f32>
      %56 = stablehlo.multiply %7, %55 : tensor<f32>
      %57 = stablehlo.add %53, %7 : tensor<f32>
      %58 = stablehlo.add %57, %54 : tensor<f32>
      %59 = stablehlo.add %58, %56 : tensor<f32>
      %60 = stablehlo.log %45 : tensor<f32>
      %61 = stablehlo.compare LT, %60, %59 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %62 = stablehlo.compare GT, %49, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %63 = stablehlo.and %61, %62 : tensor<i1>
      %64 = stablehlo.constant dense<1> : tensor<i32>
      %65 = stablehlo.add %34, %64 : tensor<i32>
      stablehlo.return %65, %63, %50 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %66, %67 = stablehlo.rng_bit_generator %26, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %68 = stablehlo.constant dense<9> : tensor<ui32>
    %69 = stablehlo.shift_right_logical %67, %68 : tensor<ui32>
    %70 = stablehlo.convert %69 : (tensor<ui32>) -> tensor<f32>
    %71 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %72 = stablehlo.multiply %70, %71 : tensor<f32>
    %75 = stablehlo.multiply %37#2, %3 : tensor<f32>
    %76 = stablehlo.divide %75, %1 : tensor<f32>
    %77, %78 = stablehlo.rng_bit_generator %66, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %79 = stablehlo.constant dense<9> : tensor<ui32>
    %80 = stablehlo.shift_right_logical %78, %79 : tensor<ui32>
    %81 = stablehlo.convert %80 : (tensor<ui32>) -> tensor<f32>
    %82 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %83 = stablehlo.multiply %81, %82 : tensor<f32>
    %84 = stablehlo.negate %76 : tensor<f32>
    %85 = stablehlo.exponential %84 : tensor<f32>
    %91:5 = stablehlo.while(%86 = %2, %87 = %85, %88 = %85, %89 = %4, %90 = %2) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %92 = stablehlo.constant dense<256.0> : tensor<f32>
      %93 = stablehlo.compare LT, %86, %92 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %94 = stablehlo.not %89 : tensor<i1>
      %95 = stablehlo.and %94, %93 : tensor<i1>
      stablehlo.return %95 : tensor<i1>
    } do {
      %96 = stablehlo.compare LE, %83, %87 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %97 = stablehlo.constant dense<1.0> : tensor<f32>
      %98 = stablehlo.add %86, %97 : tensor<f32>
      %99 = stablehlo.divide %76, %98 : tensor<f32>
      %100 = stablehlo.multiply %88, %99 : tensor<f32>
      %101 = stablehlo.add %87, %100 : tensor<f32>
      stablehlo.return %98, %101, %100, %96, %86 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %91#4, %77 : tensor<f32>, tensor<2xui64>
  }
}
