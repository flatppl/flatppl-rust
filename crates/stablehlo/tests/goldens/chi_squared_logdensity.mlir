module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.constant dense<0.5> : tensor<f32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.multiply %5, %arg0 : tensor<f32>
    %9 = stablehlo.constant dense<0.6931471805599453> : tensor<f32>
    %10 = stablehlo.multiply %8, %9 : tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = chlo.lgamma %8 : tensor<f32> -> tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.subtract %8, %7 : tensor<f32>
    %15 = stablehlo.constant dense<"0x187231BF"> : tensor<f32>
    %16 = stablehlo.multiply %14, %15 : tensor<f32>
    %19 = stablehlo.constant dense<"0x000080BE"> : tensor<f32>
    %20 = stablehlo.add %11, %13 : tensor<f32>
    %21 = stablehlo.add %20, %16 : tensor<f32>
    %22 = stablehlo.add %21, %19 : tensor<f32>
    return %22, %4 : tensor<f32>, tensor<i32>
  }
}
