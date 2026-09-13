module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<3.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.compare LT, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.constant dense<4.0> : tensor<f32>
    %5 = stablehlo.select %3, %4, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %6 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %7 = stablehlo.subtract %5, %6 : tensor<f32>
    %8 = stablehlo.constant dense<9.0> : tensor<f32>
    %9 = stablehlo.multiply %8, %7 : tensor<f32>
    %10 = stablehlo.sqrt %9 : tensor<f32>
    %11 = stablehlo.divide %1, %10 : tensor<f32>
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
    %34 = stablehlo.constant dense<false> : tensor<i1>
    %38:3 = stablehlo.while(%35 = %33, %36 = %34, %37 = %2) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %39 = stablehlo.constant dense<128> : tensor<i32>
      %40 = stablehlo.compare LT, %35, %39, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %41 = stablehlo.not %36 : tensor<i1>
      %42 = stablehlo.and %41, %40 : tensor<i1>
      stablehlo.return %42 : tensor<i1>
    } do {
      %43 = stablehlo.dynamic_slice %25, %35, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %44 = stablehlo.reshape %43 : (tensor<1xf32>) -> tensor<f32>
      %45 = stablehlo.dynamic_slice %32, %35, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %46 = stablehlo.reshape %45 : (tensor<1xf32>) -> tensor<f32>
      %47 = stablehlo.multiply %11, %44 : tensor<f32>
      %48 = stablehlo.add %1, %47 : tensor<f32>
      %49 = stablehlo.multiply %48, %48 : tensor<f32>
      %50 = stablehlo.multiply %49, %48 : tensor<f32>
      %51 = stablehlo.multiply %7, %50 : tensor<f32>
      %52 = stablehlo.constant dense<0.5> : tensor<f32>
      %53 = stablehlo.multiply %44, %44 : tensor<f32>
      %54 = stablehlo.multiply %52, %53 : tensor<f32>
      %55 = stablehlo.negate %51 : tensor<f32>
      %56 = stablehlo.log %50 : tensor<f32>
      %57 = stablehlo.multiply %7, %56 : tensor<f32>
      %58 = stablehlo.add %54, %7 : tensor<f32>
      %59 = stablehlo.add %58, %55 : tensor<f32>
      %60 = stablehlo.add %59, %57 : tensor<f32>
      %61 = stablehlo.log %46 : tensor<f32>
      %62 = stablehlo.compare LT, %61, %60 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %63 = stablehlo.compare GT, %50, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %64 = stablehlo.and %62, %63 : tensor<i1>
      %65 = stablehlo.constant dense<1> : tensor<i32>
      %66 = stablehlo.add %35, %65 : tensor<i32>
      stablehlo.return %66, %64, %51 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %67, %68 = stablehlo.rng_bit_generator %26, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %69 = stablehlo.constant dense<9> : tensor<ui32>
    %70 = stablehlo.shift_right_logical %68, %69 : tensor<ui32>
    %71 = stablehlo.convert %70 : (tensor<ui32>) -> tensor<f32>
    %72 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %73 = stablehlo.multiply %71, %72 : tensor<f32>
    %74 = stablehlo.constant dense<0.3333333432674408> : tensor<f32>
    %75 = stablehlo.power %73, %74 : tensor<f32>
    %76 = stablehlo.select %3, %75, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %77 = stablehlo.multiply %38#2, %76 : tensor<f32>
    %78 = stablehlo.divide %77, %1 : tensor<f32>
    %79 = stablehlo.divide %1, %78 : tensor<f32>
    return %79, %67 : tensor<f32>, tensor<2xui64>
  }
}
