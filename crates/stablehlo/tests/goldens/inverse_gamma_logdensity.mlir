module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.log %arg1 : tensor<f32>
    %10 = stablehlo.multiply %arg0, %9 : tensor<f32>
    %11 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.add %arg0, %8 : tensor<f32>
    %14 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %15 = stablehlo.multiply %13, %14 : tensor<f32>
    %16 = stablehlo.negate %15 : tensor<f32>
    %17 = stablehlo.divide %arg1, %6 : tensor<f32>
    %18 = stablehlo.negate %17 : tensor<f32>
    %19 = stablehlo.add %10, %12 : tensor<f32>
    %20 = stablehlo.add %19, %16 : tensor<f32>
    %21 = stablehlo.add %20, %18 : tensor<f32>
    return %21, %5 : tensor<f32>, tensor<i32>
  }
}
