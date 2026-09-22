module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<5.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.5> : tensor<f32>
    %2 = stablehlo.constant dense<2.5> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %2, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.constant dense<3.5> : tensor<f32>
    %7 = stablehlo.select %5, %6, %2 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
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
    %40:3 = stablehlo.while(%37 = %35, %38 = %36, %39 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %41 = stablehlo.constant dense<128> : tensor<i32>
      %42 = stablehlo.compare LT, %37, %41, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %43 = stablehlo.not %38 : tensor<i1>
      %44 = stablehlo.and %43, %42 : tensor<i1>
      stablehlo.return %44 : tensor<i1>
    } do {
      %45 = stablehlo.dynamic_slice %27, %37, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %46 = stablehlo.reshape %45 : (tensor<1xf32>) -> tensor<f32>
      %47 = stablehlo.dynamic_slice %34, %37, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %48 = stablehlo.reshape %47 : (tensor<1xf32>) -> tensor<f32>
      %49 = stablehlo.multiply %13, %46 : tensor<f32>
      %50 = stablehlo.add %4, %49 : tensor<f32>
      %51 = stablehlo.multiply %50, %50 : tensor<f32>
      %52 = stablehlo.multiply %51, %50 : tensor<f32>
      %53 = stablehlo.multiply %9, %52 : tensor<f32>
      %54 = stablehlo.constant dense<0.5> : tensor<f32>
      %55 = stablehlo.multiply %46, %46 : tensor<f32>
      %56 = stablehlo.multiply %54, %55 : tensor<f32>
      %57 = stablehlo.negate %53 : tensor<f32>
      %58 = stablehlo.log %52 : tensor<f32>
      %59 = stablehlo.multiply %9, %58 : tensor<f32>
      %60 = stablehlo.add %56, %9 : tensor<f32>
      %61 = stablehlo.add %60, %57 : tensor<f32>
      %62 = stablehlo.add %61, %59 : tensor<f32>
      %63 = stablehlo.log %48 : tensor<f32>
      %64 = stablehlo.compare LT, %63, %62 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %65 = stablehlo.compare GT, %52, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %66 = stablehlo.and %64, %65 : tensor<i1>
      %67 = stablehlo.constant dense<1> : tensor<i32>
      %68 = stablehlo.add %37, %67 : tensor<i32>
      stablehlo.return %68, %66, %53 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %69, %70 = stablehlo.rng_bit_generator %28, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %71 = stablehlo.constant dense<9> : tensor<ui32>
    %72 = stablehlo.shift_right_logical %70, %71 : tensor<ui32>
    %73 = stablehlo.convert %72 : (tensor<ui32>) -> tensor<f32>
    %74 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %75 = stablehlo.multiply %73, %74 : tensor<f32>
    %76 = stablehlo.constant dense<0.4000000059604645> : tensor<f32>
    %77 = stablehlo.power %75, %76 : tensor<f32>
    %78 = stablehlo.select %5, %77, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %79 = stablehlo.multiply %40#2, %78 : tensor<f32>
    %80 = stablehlo.divide %79, %1 : tensor<f32>
    %81, %82 = stablehlo.rng_bit_generator %69, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %83 = stablehlo.constant dense<9> : tensor<ui32>
    %84 = stablehlo.shift_right_logical %82, %83 : tensor<ui32>
    %85 = stablehlo.convert %84 : (tensor<ui32>) -> tensor<f32>
    %86 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %87 = stablehlo.multiply %85, %86 : tensor<f32>
    %88 = stablehlo.constant dense<2.0> : tensor<f32>
    %89 = stablehlo.multiply %87, %88 : tensor<f32>
    %90 = stablehlo.subtract %89, %4 : tensor<f32>
    %91 = chlo.erf_inv %90 : tensor<f32> -> tensor<f32>
    %92 = stablehlo.constant dense<1.4142135> : tensor<f32>
    %93 = stablehlo.multiply %91, %92 : tensor<f32>
    %94 = stablehlo.divide %80, %0 : tensor<f32>
    %95 = stablehlo.sqrt %94 : tensor<f32>
    %96 = stablehlo.divide %93, %95 : tensor<f32>
    return %96, %81 : tensor<f32>, tensor<2xui64>
  }
}
