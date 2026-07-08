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
    %28, %29 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %30 = stablehlo.constant dense<9> : tensor<128xui32>
    %31 = stablehlo.shift_right_logical %29, %30 : tensor<128xui32>
    %32 = stablehlo.convert %31 : (tensor<128xui32>) -> tensor<128xf32>
    %33 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<128xf32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.constant dense<false> : tensor<i1>
    %37 = stablehlo.constant dense<0.0> : tensor<f32>
    %41:3 = stablehlo.while(%38 = %35, %39 = %36, %40 = %37) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %42 = stablehlo.constant dense<128> : tensor<i32>
      %43 = stablehlo.compare LT, %38, %42, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %44 = stablehlo.not %39 : tensor<i1>
      %45 = stablehlo.and %44, %43 : tensor<i1>
      stablehlo.return %45 : tensor<i1>
    } do {
      %46 = stablehlo.dynamic_slice %27, %38, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %47 = stablehlo.reshape %46 : (tensor<1xf32>) -> tensor<f32>
      %48 = stablehlo.dynamic_slice %34, %38, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %49 = stablehlo.reshape %48 : (tensor<1xf32>) -> tensor<f32>
      %50 = stablehlo.multiply %13, %47 : tensor<f32>
      %51 = stablehlo.add %4, %50 : tensor<f32>
      %52 = stablehlo.multiply %51, %51 : tensor<f32>
      %53 = stablehlo.multiply %52, %51 : tensor<f32>
      %54 = stablehlo.multiply %9, %53 : tensor<f32>
      %55 = stablehlo.constant dense<0.5> : tensor<f32>
      %56 = stablehlo.multiply %47, %47 : tensor<f32>
      %57 = stablehlo.multiply %55, %56 : tensor<f32>
      %58 = stablehlo.multiply %9, %53 : tensor<f32>
      %59 = stablehlo.negate %58 : tensor<f32>
      %60 = stablehlo.log %53 : tensor<f32>
      %61 = stablehlo.multiply %9, %60 : tensor<f32>
      %62 = stablehlo.add %57, %9 : tensor<f32>
      %63 = stablehlo.add %62, %59 : tensor<f32>
      %64 = stablehlo.add %63, %61 : tensor<f32>
      %65 = stablehlo.log %49 : tensor<f32>
      %66 = stablehlo.compare LT, %65, %64 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %67 = stablehlo.compare GT, %53, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %68 = stablehlo.and %66, %67 : tensor<i1>
      %69 = stablehlo.constant dense<1> : tensor<i32>
      %70 = stablehlo.add %38, %69 : tensor<i32>
      stablehlo.return %70, %68, %54 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %71, %72 = stablehlo.rng_bit_generator %28, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %73 = stablehlo.constant dense<9> : tensor<ui32>
    %74 = stablehlo.shift_right_logical %72, %73 : tensor<ui32>
    %75 = stablehlo.convert %74 : (tensor<ui32>) -> tensor<f32>
    %76 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %77 = stablehlo.multiply %75, %76 : tensor<f32>
    %78 = stablehlo.divide %4, %1 : tensor<f32>
    %79 = stablehlo.power %77, %78 : tensor<f32>
    %80 = stablehlo.select %5, %79, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %81 = stablehlo.multiply %41#2, %80 : tensor<f32>
    %82 = stablehlo.divide %81, %2 : tensor<f32>
    %83, %84 = stablehlo.rng_bit_generator %71, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %85 = stablehlo.constant dense<9> : tensor<ui32>
    %86 = stablehlo.shift_right_logical %84, %85 : tensor<ui32>
    %87 = stablehlo.convert %86 : (tensor<ui32>) -> tensor<f32>
    %88 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %89 = stablehlo.multiply %87, %88 : tensor<f32>
    %90 = stablehlo.negate %82 : tensor<f32>
    %91 = stablehlo.exponential %90 : tensor<f32>
    %92 = stablehlo.constant dense<0.0> : tensor<f32>
    %93 = stablehlo.constant dense<false> : tensor<i1>
    %94 = stablehlo.constant dense<0.0> : tensor<f32>
    %100:5 = stablehlo.while(%95 = %92, %96 = %91, %97 = %91, %98 = %93, %99 = %94) : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    cond {
      %101 = stablehlo.constant dense<256.0> : tensor<f32>
      %102 = stablehlo.compare LT, %95, %101 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %103 = stablehlo.not %98 : tensor<i1>
      %104 = stablehlo.and %103, %102 : tensor<i1>
      stablehlo.return %104 : tensor<i1>
    } do {
      %105 = stablehlo.compare LE, %89, %96 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %106 = stablehlo.constant dense<1.0> : tensor<f32>
      %107 = stablehlo.add %95, %106 : tensor<f32>
      %108 = stablehlo.divide %82, %107 : tensor<f32>
      %109 = stablehlo.multiply %97, %108 : tensor<f32>
      %110 = stablehlo.add %96, %109 : tensor<f32>
      stablehlo.return %107, %110, %109, %105, %95 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<i1>, tensor<f32>
    }
    return %100#4, %83 : tensor<f32>, tensor<2xui64>
  }
}
