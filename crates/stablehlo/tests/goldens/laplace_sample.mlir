module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4, %5 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %6 = stablehlo.constant dense<9> : tensor<ui32>
    %7 = stablehlo.shift_right_logical %5, %6 : tensor<ui32>
    %8 = stablehlo.convert %7 : (tensor<ui32>) -> tensor<f32>
    %9 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %10 = stablehlo.multiply %8, %9 : tensor<f32>
    %11 = stablehlo.constant dense<0.5> : tensor<f32>
    %12 = stablehlo.subtract %10, %11 : tensor<f32>
    %13 = stablehlo.compare GE, %12, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %14 = stablehlo.constant dense<1.0> : tensor<f32>
    %15 = stablehlo.constant dense<-1.0> : tensor<f32>
    %16 = stablehlo.select %13, %14, %15 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %17 = stablehlo.abs %12 : tensor<f32>
    %18 = stablehlo.constant dense<2.0> : tensor<f32>
    %19 = stablehlo.multiply %18, %17 : tensor<f32>
    %20 = stablehlo.subtract %3, %19 : tensor<f32>
    %21 = stablehlo.log %20 : tensor<f32>
    %22 = stablehlo.multiply %16, %21 : tensor<f32>
    %23 = stablehlo.multiply %1, %22 : tensor<f32>
    %24 = stablehlo.negate %23 : tensor<f32>
    %25 = stablehlo.add %0, %24 : tensor<f32>
    return %25, %4 : tensor<f32>, tensor<2xui64>
  }
}
