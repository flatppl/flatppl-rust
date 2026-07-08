module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
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
    %36 = stablehlo.constant dense<0.0> : tensor<f32>
    %40:3 = stablehlo.while(%37 = %34, %38 = %35, %39 = %36) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %41 = stablehlo.constant dense<128> : tensor<i32>
      %42 = stablehlo.compare LT, %37, %41, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %43 = stablehlo.not %38 : tensor<i1>
      %44 = stablehlo.and %43, %42 : tensor<i1>
      stablehlo.return %44 : tensor<i1>
    } do {
      %45 = stablehlo.dynamic_slice %26, %37, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %46 = stablehlo.reshape %45 : (tensor<1xf32>) -> tensor<f32>
      %47 = stablehlo.dynamic_slice %33, %37, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %48 = stablehlo.reshape %47 : (tensor<1xf32>) -> tensor<f32>
      %49 = stablehlo.multiply %12, %46 : tensor<f32>
      %50 = stablehlo.add %3, %49 : tensor<f32>
      %51 = stablehlo.multiply %50, %50 : tensor<f32>
      %52 = stablehlo.multiply %51, %50 : tensor<f32>
      %53 = stablehlo.multiply %8, %52 : tensor<f32>
      %54 = stablehlo.constant dense<0.5> : tensor<f32>
      %55 = stablehlo.multiply %46, %46 : tensor<f32>
      %56 = stablehlo.multiply %54, %55 : tensor<f32>
      %57 = stablehlo.multiply %8, %52 : tensor<f32>
      %58 = stablehlo.negate %57 : tensor<f32>
      %59 = stablehlo.log %52 : tensor<f32>
      %60 = stablehlo.multiply %8, %59 : tensor<f32>
      %61 = stablehlo.add %56, %8 : tensor<f32>
      %62 = stablehlo.add %61, %58 : tensor<f32>
      %63 = stablehlo.add %62, %60 : tensor<f32>
      %64 = stablehlo.log %48 : tensor<f32>
      %65 = stablehlo.compare LT, %64, %63 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %66 = stablehlo.compare GT, %52, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %67 = stablehlo.and %65, %66 : tensor<i1>
      %68 = stablehlo.constant dense<1> : tensor<i32>
      %69 = stablehlo.add %37, %68 : tensor<i32>
      stablehlo.return %69, %67, %53 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %70, %71 = stablehlo.rng_bit_generator %27, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %72 = stablehlo.constant dense<9> : tensor<ui32>
    %73 = stablehlo.shift_right_logical %71, %72 : tensor<ui32>
    %74 = stablehlo.convert %73 : (tensor<ui32>) -> tensor<f32>
    %75 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %76 = stablehlo.multiply %74, %75 : tensor<f32>
    %77 = stablehlo.divide %3, %0 : tensor<f32>
    %78 = stablehlo.power %76, %77 : tensor<f32>
    %79 = stablehlo.select %4, %78, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %80 = stablehlo.multiply %40#2, %79 : tensor<f32>
    %81 = stablehlo.divide %80, %1 : tensor<f32>
    %82, %83 = stablehlo.rng_bit_generator %70, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %84 = stablehlo.constant dense<9> : tensor<ui32>
    %85 = stablehlo.shift_right_logical %83, %84 : tensor<ui32>
    %86 = stablehlo.convert %85 : (tensor<ui32>) -> tensor<f32>
    %87 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %88 = stablehlo.multiply %86, %87 : tensor<f32>
    %89 = stablehlo.negate %81 : tensor<f32>
    %90 = stablehlo.exponential %89 : tensor<f32>
    %91 = stablehlo.constant dense<0.0> : tensor<f32>
    %92 = stablehlo.constant dense<false> : tensor<i1>
    %93 = stablehlo.constant dense<0.0> : tensor<f32>
    %99:5 = stablehlo.while(%94 = %91, %95 = %90, %96 = %90, %97 = %92, %98 = %93) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %100 = stablehlo.constant dense<256.0> : tensor<f32>
      %101 = stablehlo.compare LT, %94, %100 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %102 = stablehlo.not %97 : tensor<i1>
      %103 = stablehlo.and %102, %101 : tensor<i1>
      stablehlo.return %103 : tensor<i1>
    } do {
      %104 = stablehlo.compare LE, %88, %95 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %105 = stablehlo.constant dense<1.0> : tensor<f32>
      %106 = stablehlo.add %94, %105 : tensor<f32>
      %107 = stablehlo.divide %81, %106 : tensor<f32>
      %108 = stablehlo.multiply %96, %107 : tensor<f32>
      %109 = stablehlo.add %95, %108 : tensor<f32>
      stablehlo.return %106, %109, %108, %104, %94 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %99#4, %82 : tensor<f32>, tensor<2xui64>
  }
}
