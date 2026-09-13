module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
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
    %35 = stablehlo.constant dense<false> : tensor<i1>
    %39:3 = stablehlo.while(%36 = %34, %37 = %35, %38 = %2) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %40 = stablehlo.constant dense<128> : tensor<i32>
      %41 = stablehlo.compare LT, %36, %40, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %42 = stablehlo.not %37 : tensor<i1>
      %43 = stablehlo.and %42, %41 : tensor<i1>
      stablehlo.return %43 : tensor<i1>
    } do {
      %44 = stablehlo.dynamic_slice %26, %36, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %45 = stablehlo.reshape %44 : (tensor<1xf32>) -> tensor<f32>
      %46 = stablehlo.dynamic_slice %33, %36, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %47 = stablehlo.reshape %46 : (tensor<1xf32>) -> tensor<f32>
      %48 = stablehlo.multiply %12, %45 : tensor<f32>
      %49 = stablehlo.add %3, %48 : tensor<f32>
      %50 = stablehlo.multiply %49, %49 : tensor<f32>
      %51 = stablehlo.multiply %50, %49 : tensor<f32>
      %52 = stablehlo.multiply %8, %51 : tensor<f32>
      %53 = stablehlo.constant dense<0.5> : tensor<f32>
      %54 = stablehlo.multiply %45, %45 : tensor<f32>
      %55 = stablehlo.multiply %53, %54 : tensor<f32>
      %56 = stablehlo.negate %52 : tensor<f32>
      %57 = stablehlo.log %51 : tensor<f32>
      %58 = stablehlo.multiply %8, %57 : tensor<f32>
      %59 = stablehlo.add %55, %8 : tensor<f32>
      %60 = stablehlo.add %59, %56 : tensor<f32>
      %61 = stablehlo.add %60, %58 : tensor<f32>
      %62 = stablehlo.log %47 : tensor<f32>
      %63 = stablehlo.compare LT, %62, %61 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %64 = stablehlo.compare GT, %51, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %65 = stablehlo.and %63, %64 : tensor<i1>
      %66 = stablehlo.constant dense<1> : tensor<i32>
      %67 = stablehlo.add %36, %66 : tensor<i32>
      stablehlo.return %67, %65, %52 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %68, %69 = stablehlo.rng_bit_generator %27, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %70 = stablehlo.constant dense<9> : tensor<ui32>
    %71 = stablehlo.shift_right_logical %69, %70 : tensor<ui32>
    %72 = stablehlo.convert %71 : (tensor<ui32>) -> tensor<f32>
    %73 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %74 = stablehlo.multiply %72, %73 : tensor<f32>
    %75 = stablehlo.constant dense<0.20000000298023224> : tensor<f32>
    %76 = stablehlo.power %74, %75 : tensor<f32>
    %77 = stablehlo.select %4, %76, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %78 = stablehlo.multiply %39#2, %77 : tensor<f32>
    %79 = stablehlo.divide %78, %1 : tensor<f32>
    %80, %81 = stablehlo.rng_bit_generator %68, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %82 = stablehlo.constant dense<9> : tensor<ui32>
    %83 = stablehlo.shift_right_logical %81, %82 : tensor<ui32>
    %84 = stablehlo.convert %83 : (tensor<ui32>) -> tensor<f32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %86 = stablehlo.multiply %84, %85 : tensor<f32>
    %87 = stablehlo.negate %79 : tensor<f32>
    %88 = stablehlo.exponential %87 : tensor<f32>
    %94:5 = stablehlo.while(%89 = %2, %90 = %88, %91 = %88, %92 = %35, %93 = %2) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %95 = stablehlo.constant dense<256.0> : tensor<f32>
      %96 = stablehlo.compare LT, %89, %95 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %97 = stablehlo.not %92 : tensor<i1>
      %98 = stablehlo.and %97, %96 : tensor<i1>
      stablehlo.return %98 : tensor<i1>
    } do {
      %99 = stablehlo.compare LE, %86, %90 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %100 = stablehlo.constant dense<1.0> : tensor<f32>
      %101 = stablehlo.add %89, %100 : tensor<f32>
      %102 = stablehlo.divide %79, %101 : tensor<f32>
      %103 = stablehlo.multiply %91, %102 : tensor<f32>
      %104 = stablehlo.add %90, %103 : tensor<f32>
      stablehlo.return %101, %104, %103, %99, %89 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %94#4, %80 : tensor<f32>, tensor<2xui64>
  }
}
