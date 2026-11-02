module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.select %1, %1, %2 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<9> : tensor<i32>
    %6 = stablehlo.select %3, %5, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = stablehlo.constant dense<0.5> : tensor<f32>
    %9 = stablehlo.constant dense<1.0> : tensor<f32>
    %10 = stablehlo.log %arg1 : tensor<f32>
    %11 = stablehlo.multiply %arg0, %10 : tensor<f32>
    %12 = chlo.lgamma %arg0 : tensor<f32> -> tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.add %arg0, %9 : tensor<f32>
    %15 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %16 = stablehlo.multiply %14, %15 : tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = stablehlo.divide %arg1, %7 : tensor<f32>
    %19 = stablehlo.negate %18 : tensor<f32>
    %20 = stablehlo.add %11, %13 : tensor<f32>
    %21 = stablehlo.add %20, %17 : tensor<f32>
    %22 = stablehlo.add %21, %19 : tensor<f32>
    return %22, %6 : tensor<f32>, tensor<i32>
  }
}
