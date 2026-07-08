module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.2> : tensor<f32>
    %1 = stablehlo.constant dense<0.3> : tensor<f32>
    %2 = stablehlo.constant dense<0.5> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<0.0> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9, %10 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %11 = stablehlo.constant dense<9> : tensor<ui32>
    %12 = stablehlo.shift_right_logical %10, %11 : tensor<ui32>
    %13 = stablehlo.convert %12 : (tensor<ui32>) -> tensor<f32>
    %14 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %15 = stablehlo.multiply %13, %14 : tensor<f32>
    %16 = stablehlo.constant dense<0.0> : tensor<f32>
    %17 = stablehlo.constant dense<0.0> : tensor<f32>
    %18 = stablehlo.slice %6 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %19 = stablehlo.reshape %18 : (tensor<1xf32>) -> tensor<f32>
    %20 = stablehlo.add %16, %19 : tensor<f32>
    %21 = stablehlo.compare LT, %20, %15 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %22 = stablehlo.select %21, %8, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %23 = stablehlo.add %17, %22 : tensor<f32>
    %24 = stablehlo.slice %6 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %25 = stablehlo.reshape %24 : (tensor<1xf32>) -> tensor<f32>
    %26 = stablehlo.add %20, %25 : tensor<f32>
    %27 = stablehlo.compare LT, %26, %15 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %28 = stablehlo.select %27, %8, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %29 = stablehlo.add %23, %28 : tensor<f32>
    return %29, %9 : tensor<f32>, tensor<2xui64>
  }
}
