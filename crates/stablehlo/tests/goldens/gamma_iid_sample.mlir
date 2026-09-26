module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %6 = stablehlo.constant dense<1.6666666269302368> : tensor<f32>
    %10 = stablehlo.constant dense<0.25819888710975647> : tensor<f32>
    %11, %12 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %13 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %14 = stablehlo.shift_right_logical %12, %13 : tensor<128x4xui32>
    %15 = stablehlo.convert %14 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %16 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<128x4xf32>
    %18 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %19 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %20 = stablehlo.multiply %17, %18 : tensor<128x4xf32>
    %21 = stablehlo.subtract %20, %19 : tensor<128x4xf32>
    %22 = chlo.erf_inv %21 : tensor<128x4xf32> -> tensor<128x4xf32>
    %23 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128x4xf32>
    %25, %26 = stablehlo.rng_bit_generator %11, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %27 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %28 = stablehlo.shift_right_logical %26, %27 : tensor<128x4xui32>
    %29 = stablehlo.convert %28 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %30 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<128x4xf32>
    %32 = stablehlo.constant dense<0> : tensor<i32>
    %33 = stablehlo.constant dense<false> : tensor<4xi1>
    %34 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %38:3 = stablehlo.while(%35 = %32, %36 = %33, %37 = %34) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %39 = stablehlo.constant dense<128> : tensor<i32>
      %40 = stablehlo.compare LT, %35, %39, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %41 = stablehlo.constant dense<true> : tensor<i1>
      %42 = stablehlo.reduce(%36 init: %41) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %43 = stablehlo.not %42 : tensor<i1>
      %44 = stablehlo.and %40, %43 : tensor<i1>
      stablehlo.return %44 : tensor<i1>
    } do {
      %45 = stablehlo.constant dense<0> : tensor<i32>
      %46 = stablehlo.dynamic_slice %24, %35, %45, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %47 = stablehlo.reshape %46 : (tensor<1x4xf32>) -> tensor<4xf32>
      %48 = stablehlo.dynamic_slice %31, %35, %45, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %49 = stablehlo.reshape %48 : (tensor<1x4xf32>) -> tensor<4xf32>
      %50 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %51 = stablehlo.multiply %50, %47 : tensor<4xf32>
      %52 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %53 = stablehlo.add %52, %51 : tensor<4xf32>
      %54 = stablehlo.multiply %53, %53 : tensor<4xf32>
      %55 = stablehlo.multiply %54, %53 : tensor<4xf32>
      %56 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %57 = stablehlo.multiply %56, %55 : tensor<4xf32>
      %58 = stablehlo.constant dense<0.5> : tensor<f32>
      %59 = stablehlo.multiply %47, %47 : tensor<4xf32>
      %60 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %61 = stablehlo.multiply %60, %59 : tensor<4xf32>
      %62 = stablehlo.negate %57 : tensor<4xf32>
      %63 = stablehlo.log %55 : tensor<4xf32>
      %64 = stablehlo.multiply %56, %63 : tensor<4xf32>
      %65 = stablehlo.add %61, %56 : tensor<4xf32>
      %66 = stablehlo.add %65, %62 : tensor<4xf32>
      %67 = stablehlo.add %66, %64 : tensor<4xf32>
      %68 = stablehlo.log %49 : tensor<4xf32>
      %69 = stablehlo.compare LT, %68, %67 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %70 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %71 = stablehlo.compare GT, %55, %70 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %72 = stablehlo.and %69, %71 : tensor<4xi1>
      %73 = stablehlo.select %36, %37, %57 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %74 = stablehlo.or %36, %72 : tensor<4xi1>
      %75 = stablehlo.constant dense<1> : tensor<i32>
      %76 = stablehlo.add %35, %75 : tensor<i32>
      stablehlo.return %76, %74, %73 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %77, %78 = stablehlo.rng_bit_generator %25, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %79 = stablehlo.constant dense<9> : tensor<4xui32>
    %80 = stablehlo.shift_right_logical %78, %79 : tensor<4xui32>
    %81 = stablehlo.convert %80 : (tensor<4xui32>) -> tensor<4xf32>
    %82 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %83 = stablehlo.multiply %81, %82 : tensor<4xf32>
    %87 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %88 = stablehlo.multiply %38#2, %87 : tensor<4xf32>
    %89 = stablehlo.divide %88, %87 : tensor<4xf32>
    return %89, %77 : tensor<4xf32>, tensor<2xui64>
  }
}
