module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LT, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.compare GT, %arg0, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.select %1, %1, %3 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<9> : tensor<i32>
    %7 = stablehlo.select %4, %6, %5 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = stablehlo.log %arg0 : tensor<f32>
    %10 = stablehlo.negate %arg0 : tensor<f32>
    %11 = stablehlo.log_plus_one %10 : tensor<f32>
    %12 = stablehlo.constant dense<"0x00008040"> : tensor<f32>
    %13 = stablehlo.multiply %12, %11 : tensor<f32>
    %14 = stablehlo.add %9, %13 : tensor<f32>
    return %14, %7 : tensor<f32>, tensor<i32>
  }
}
