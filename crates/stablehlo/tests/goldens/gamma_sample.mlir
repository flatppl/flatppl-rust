module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<false> : tensor<i1>
    %6 = stablehlo.constant dense<"0x5555D53F"> : tensor<f32>
    %10 = stablehlo.constant dense<"0xA532843E"> : tensor<f32>
    %11, %12 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %13 = stablehlo.constant dense<9> : tensor<128xui32>
    %14 = stablehlo.shift_right_logical %12, %13 : tensor<128xui32>
    %15 = stablehlo.convert %14 : (tensor<128xui32>) -> tensor<128xf32>
    %16 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<128xf32>
    %18 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %19 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %20 = stablehlo.multiply %17, %18 : tensor<128xf32>
    %21 = stablehlo.subtract %20, %19 : tensor<128xf32>
    %22 = chlo.erf_inv %21 : tensor<128xf32> -> tensor<128xf32>
    %23 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128xf32>
    %25, %26 = stablehlo.rng_bit_generator %11, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %27 = stablehlo.constant dense<9> : tensor<128xui32>
    %28 = stablehlo.shift_right_logical %26, %27 : tensor<128xui32>
    %29 = stablehlo.convert %28 : (tensor<128xui32>) -> tensor<128xf32>
    %30 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<128xf32>
    %32 = stablehlo.constant dense<0> : tensor<i32>
    %36:3 = stablehlo.while(%33 = %32, %34 = %3, %35 = %2) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %37 = stablehlo.constant dense<128> : tensor<i32>
      %38 = stablehlo.compare LT, %33, %37, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %39 = stablehlo.not %34 : tensor<i1>
      %40 = stablehlo.and %39, %38 : tensor<i1>
      stablehlo.return %40 : tensor<i1>
    } do {
      %41 = stablehlo.dynamic_slice %24, %33, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %42 = stablehlo.reshape %41 : (tensor<1xf32>) -> tensor<f32>
      %43 = stablehlo.dynamic_slice %31, %33, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %44 = stablehlo.reshape %43 : (tensor<1xf32>) -> tensor<f32>
      %45 = stablehlo.multiply %10, %42 : tensor<f32>
      %46 = stablehlo.add %1, %45 : tensor<f32>
      %47 = stablehlo.multiply %46, %46 : tensor<f32>
      %48 = stablehlo.multiply %47, %46 : tensor<f32>
      %49 = stablehlo.multiply %6, %48 : tensor<f32>
      %50 = stablehlo.constant dense<0.5> : tensor<f32>
      %51 = stablehlo.multiply %42, %42 : tensor<f32>
      %52 = stablehlo.multiply %50, %51 : tensor<f32>
      %53 = stablehlo.negate %49 : tensor<f32>
      %54 = stablehlo.log %48 : tensor<f32>
      %55 = stablehlo.multiply %6, %54 : tensor<f32>
      %56 = stablehlo.add %52, %6 : tensor<f32>
      %57 = stablehlo.add %56, %53 : tensor<f32>
      %58 = stablehlo.add %57, %55 : tensor<f32>
      %59 = stablehlo.log %44 : tensor<f32>
      %60 = stablehlo.compare LT, %59, %58 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %61 = stablehlo.compare GT, %48, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %62 = stablehlo.and %60, %61 : tensor<i1>
      %63 = stablehlo.constant dense<1> : tensor<i32>
      %64 = stablehlo.add %33, %63 : tensor<i32>
      stablehlo.return %64, %62, %49 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %65, %66 = stablehlo.rng_bit_generator %25, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %67 = stablehlo.constant dense<9> : tensor<ui32>
    %68 = stablehlo.shift_right_logical %66, %67 : tensor<ui32>
    %69 = stablehlo.convert %68 : (tensor<ui32>) -> tensor<f32>
    %70 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %71 = stablehlo.multiply %69, %70 : tensor<f32>
    %74 = stablehlo.multiply %36#2, %1 : tensor<f32>
    %75 = stablehlo.divide %74, %1 : tensor<f32>
    return %75, %65 : tensor<f32>, tensor<2xui64>
  }
}
