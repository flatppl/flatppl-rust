module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.constant dense<0.5> : tensor<f32>
    %2 = stablehlo.multiply %1, %0 : tensor<f32>
    %3 = stablehlo.constant dense<0.5> : tensor<f32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %5 = stablehlo.constant dense<1.0> : tensor<f32>
    %6 = stablehlo.compare LT, %2, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = stablehlo.add %2, %5 : tensor<f32>
    %8 = stablehlo.select %6, %7, %2 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %10 = stablehlo.subtract %8, %9 : tensor<f32>
    %11 = stablehlo.constant dense<9.0> : tensor<f32>
    %12 = stablehlo.multiply %11, %10 : tensor<f32>
    %13 = stablehlo.sqrt %12 : tensor<f32>
    %14 = stablehlo.divide %5, %13 : tensor<f32>
    %15, %16 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %17 = stablehlo.constant dense<9> : tensor<128xui32>
    %18 = stablehlo.shift_right_logical %16, %17 : tensor<128xui32>
    %19 = stablehlo.convert %18 : (tensor<128xui32>) -> tensor<128xf32>
    %20 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<128xf32>
    %22 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %23 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %24 = stablehlo.multiply %21, %22 : tensor<128xf32>
    %25 = stablehlo.subtract %24, %23 : tensor<128xf32>
    %26 = chlo.erf_inv %25 : tensor<128xf32> -> tensor<128xf32>
    %27 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %28 = stablehlo.multiply %26, %27 : tensor<128xf32>
    %29, %30 = stablehlo.rng_bit_generator %15, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %31 = stablehlo.constant dense<9> : tensor<128xui32>
    %32 = stablehlo.shift_right_logical %30, %31 : tensor<128xui32>
    %33 = stablehlo.convert %32 : (tensor<128xui32>) -> tensor<128xf32>
    %34 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %35 = stablehlo.multiply %33, %34 : tensor<128xf32>
    %36 = stablehlo.constant dense<0> : tensor<i32>
    %37 = stablehlo.constant dense<false> : tensor<i1>
    %38 = stablehlo.constant dense<0.0> : tensor<f32>
    %42:3 = stablehlo.while(%39 = %36, %40 = %37, %41 = %38) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %43 = stablehlo.constant dense<128> : tensor<i32>
      %44 = stablehlo.compare LT, %39, %43, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %45 = stablehlo.not %40 : tensor<i1>
      %46 = stablehlo.and %45, %44 : tensor<i1>
      stablehlo.return %46 : tensor<i1>
    } do {
      %47 = stablehlo.dynamic_slice %28, %39, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %48 = stablehlo.reshape %47 : (tensor<1xf32>) -> tensor<f32>
      %49 = stablehlo.dynamic_slice %35, %39, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %50 = stablehlo.reshape %49 : (tensor<1xf32>) -> tensor<f32>
      %51 = stablehlo.multiply %14, %48 : tensor<f32>
      %52 = stablehlo.add %5, %51 : tensor<f32>
      %53 = stablehlo.multiply %52, %52 : tensor<f32>
      %54 = stablehlo.multiply %53, %52 : tensor<f32>
      %55 = stablehlo.multiply %10, %54 : tensor<f32>
      %56 = stablehlo.constant dense<0.5> : tensor<f32>
      %57 = stablehlo.multiply %48, %48 : tensor<f32>
      %58 = stablehlo.multiply %56, %57 : tensor<f32>
      %59 = stablehlo.multiply %10, %54 : tensor<f32>
      %60 = stablehlo.negate %59 : tensor<f32>
      %61 = stablehlo.log %54 : tensor<f32>
      %62 = stablehlo.multiply %10, %61 : tensor<f32>
      %63 = stablehlo.add %58, %10 : tensor<f32>
      %64 = stablehlo.add %63, %60 : tensor<f32>
      %65 = stablehlo.add %64, %62 : tensor<f32>
      %66 = stablehlo.log %50 : tensor<f32>
      %67 = stablehlo.compare LT, %66, %65 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %68 = stablehlo.compare GT, %54, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %69 = stablehlo.and %67, %68 : tensor<i1>
      %70 = stablehlo.constant dense<1> : tensor<i32>
      %71 = stablehlo.add %39, %70 : tensor<i32>
      stablehlo.return %71, %69, %55 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %72, %73 = stablehlo.rng_bit_generator %29, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %74 = stablehlo.constant dense<9> : tensor<ui32>
    %75 = stablehlo.shift_right_logical %73, %74 : tensor<ui32>
    %76 = stablehlo.convert %75 : (tensor<ui32>) -> tensor<f32>
    %77 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %78 = stablehlo.multiply %76, %77 : tensor<f32>
    %79 = stablehlo.divide %5, %2 : tensor<f32>
    %80 = stablehlo.power %78, %79 : tensor<f32>
    %81 = stablehlo.select %6, %80, %5 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %82 = stablehlo.multiply %42#2, %81 : tensor<f32>
    %83 = stablehlo.divide %82, %3 : tensor<f32>
    return %83, %72 : tensor<f32>, tensor<2xui64>
  }
}
